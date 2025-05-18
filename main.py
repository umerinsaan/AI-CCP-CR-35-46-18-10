from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np
import os
import moviepy as mp
from pydub import AudioSegment
import matplotlib.pyplot as plt

# Step 1: Load reference voice(s)
def get_reference_embedding(ref_paths):
    encoder = VoiceEncoder()
    embeddings = []
    for path in ref_paths:
        wav = preprocess_wav(Path(path))
        emb = encoder.embed_utterance(wav)
        embeddings.append(emb)
    return np.mean(embeddings, axis=0)

# Step 2: Split target audio into chunks
def split_audio(path, chunk_length_ms=2000):
    audio = AudioSegment.from_file(path)
    chunks = [audio[i:i+chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
    return chunks

# Step 3: Compare chunks
def analyze_audio(target_path, ref_embedding, threshold=0.75):
    encoder = VoiceEncoder()
    chunks = split_audio(target_path)
    matches = []

    for i, chunk in enumerate(chunks):
        chunk_path = f"temp_chunk.wav"
        chunk.export(chunk_path, format="wav")
        wav = preprocess_wav(chunk_path)
        emb = encoder.embed_utterance(wav)
        similarity = np.dot(ref_embedding, emb) / (np.linalg.norm(ref_embedding) * np.linalg.norm(emb))
        if similarity > threshold:
            time_start = i * 2  # seconds
            matches.append(time_start)
    return matches

# Step 4: Plotting function
def plot_match_frequency(matches, chunk_length=2, audio_duration=None, bin_size=10):
    if not matches:
        print("No matches found to plot.")
        return

    import matplotlib.pyplot as plt
    import numpy as np

    # Bin the matches
    max_time = audio_duration if audio_duration else max(matches) + chunk_length
    bins = np.arange(0, max_time + bin_size, bin_size)
    counts, _ = np.histogram(matches, bins=bins)

    # Plot
    plt.figure(figsize=(14, 4))
    bar_positions = bins[:-1] + bin_size / 2

    plt.bar(bar_positions, counts, width=bin_size * 0.9, color="#1f77b4", edgecolor='black')

    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("📊 Speaker Match Frequency Over Time", fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.4)

    # Optional: Set x-axis ticks at bin edges
    plt.xticks(np.arange(0, max_time + 1, bin_size))

    plt.tight_layout()
    plt.show()


# --------------------------
# MAIN LOGIC
# --------------------------
ref_paths = ["samples/rayyan.waptt.opus"]
target_path = "video.mp4"

# Extract audio from video if needed
if target_path.endswith(".mp4"):
    video = mp.VideoFileClip(target_path)
    video.audio.write_audiofile("extracted_audio.wav")
    target_path = "extracted_audio.wav"

# Get reference embedding
ref_embedding = get_reference_embedding(ref_paths)

# Analyze audio
matches = analyze_audio(target_path, ref_embedding)

# Print and plot
print("Speaker found at seconds:", matches)

# Get audio duration
audio = AudioSegment.from_file(target_path)
duration_sec = len(audio) / 1000

# Call plot
plot_match_frequency(matches, audio_duration=duration_sec)

