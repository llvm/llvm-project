#!/usr/bin/env python3
"""
Transformer-based RAG Upgrade for LAT5150DRVMIL
Uses HuggingFace transformers for semantic embeddings

This upgrades from TF-IDF (51.8% accuracy) to transformer embeddings (target: 88%+)
Based on Maharana et al. research: BAAI/bge-base-en-v1.5 recommended
"""

import os
import json
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Set
import warnings
warnings.filterwarnings('ignore')

# Check if transformers is installed
try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  sentence-transformers not installed!")
    print("\nInstall with:")
    print("  pip install sentence-transformers")
    print("\nThis will install HuggingFace transformers and dependencies.")
    import sys
    sys.exit(1)


class TransformerRetriever:
    """
    Semantic retriever using HuggingFace transformer embeddings

    Model: BAAI/bge-base-en-v1.5 (recommended by Maharana et al.)
    - 109M parameters
    - 768-dimensional embeddings
    - State-of-art for retrieval tasks
    - Works on CPU (no GPU required)
    """

    def __init__(self, chunks: List[Dict], model_name: str = 'BAAI/bge-base-en-v1.5', use_incremental: bool = True, embedding_path: str = 'rag_system/transformer_embeddings.npz'):
        """
        Initialize transformer-based retriever

        Args:
            chunks: List of document chunks with text and metadata
            model_name: HuggingFace model identifier
            use_incremental: Use incremental updates (default: True, much faster)
            embedding_path: Path to save/load embeddings
        """
        self.chunks = chunks
        self.model_name = model_name
        self.embedding_path = embedding_path

        print(f"Loading transformer model: {model_name}")
        print("(First run may take 1-2 minutes to download ~400MB model)")

        # Load model
        self.model = SentenceTransformer(model_name)

        print(f"✓ Model loaded: {self.model.get_sentence_embedding_dimension()}-dim embeddings")

        # Build embeddings (incremental by default)
        self.chunk_embeddings = None
        if use_incremental:
            num_new = self.incremental_update(self.embedding_path)
            if num_new == 0:
                pass  # Already printed message in incremental_update
            elif num_new == len(chunks):
                pass  # Was a full build
        else:
            self._build_embeddings()

    def _build_embeddings(self):
        """Generate embeddings for all chunks"""
        print(f"\nGenerating embeddings for {len(self.chunks)} chunks...")

        # Extract text from chunks
        texts = [chunk['text'] for chunk in self.chunks]

        # Generate embeddings (batched for efficiency)
        self.chunk_embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        print(f"✓ Generated embeddings: shape {self.chunk_embeddings.shape}")

    def search(self, query: str, top_k: int = 3) -> List[Tuple[Dict, float]]:
        """
        Semantic search using cosine similarity

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of (chunk, similarity_score) tuples
        """
        # Encode query
        query_embedding = self.model.encode(query, convert_to_numpy=True)

        # Compute cosine similarities
        similarities = np.dot(self.chunk_embeddings, query_embedding) / (
            np.linalg.norm(self.chunk_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Return chunks with scores
        results = [
            (self.chunks[idx], float(similarities[idx]))
            for idx in top_indices
        ]

        return results

    @staticmethod
    def _compute_chunk_hash(chunk: Dict) -> str:
        """Compute hash of chunk content for change detection"""
        content = chunk['text'] + str(chunk['metadata'])
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def save_embeddings(self, output_path: str = None):
        """Save embeddings to disk with chunk hashes for incremental updates"""
        if output_path is None:
            output_path = self.embedding_path

        # Compute hashes for all chunks
        chunk_hashes = [self._compute_chunk_hash(chunk) for chunk in self.chunks]

        np.savez_compressed(
            output_path,
            embeddings=self.chunk_embeddings,
            chunk_hashes=chunk_hashes,
            model_name=self.model_name
        )
        print(f"✓ Saved {len(chunk_hashes)} embeddings to {output_path}")

    def incremental_update(self, embedding_path: str = 'rag_system/transformer_embeddings.npz'):
        """
        Incrementally update embeddings - only regenerate for new/changed chunks

        Returns:
            Number of new embeddings generated
        """
        # Compute current hashes
        current_hashes = [self._compute_chunk_hash(chunk) for chunk in self.chunks]

        # Try to load previous embeddings
        if not Path(embedding_path).exists():
            print("No previous embeddings found, generating all...")
            return len(self.chunks)

        try:
            data = np.load(embedding_path, allow_pickle=True)
            previous_embeddings = data['embeddings']
            previous_hashes = list(data['chunk_hashes'])

            # Find which chunks changed
            previous_hash_set = set(previous_hashes)
            new_indices = []
            new_chunks = []

            for i, (chunk, chunk_hash) in enumerate(zip(self.chunks, current_hashes)):
                if chunk_hash not in previous_hash_set:
                    new_indices.append(i)
                    new_chunks.append(chunk)

            if not new_chunks:
                print("✓ No changes detected, reusing existing embeddings")
                self.chunk_embeddings = previous_embeddings[:len(self.chunks)]
                return 0

            print(f"Detected {len(new_chunks)} new/changed chunks (out of {len(self.chunks)} total)")
            print(f"Generating embeddings for changed chunks only...")

            # Generate embeddings only for new chunks
            new_texts = [chunk['text'] for chunk in new_chunks]
            new_embeddings = self.model.encode(
                new_texts,
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True
            )

            # Build hash->embedding map from previous data
            hash_to_embedding = {}
            for prev_hash, prev_emb in zip(previous_hashes, previous_embeddings):
                hash_to_embedding[prev_hash] = prev_emb

            # Add new embeddings
            for new_hash, new_emb in zip([current_hashes[i] for i in new_indices], new_embeddings):
                hash_to_embedding[new_hash] = new_emb

            # Reconstruct embeddings in current chunk order
            self.chunk_embeddings = np.array([
                hash_to_embedding[chunk_hash]
                for chunk_hash in current_hashes
            ])

            print(f"✓ Updated embeddings: {len(new_chunks)} new, {len(self.chunks) - len(new_chunks)} reused")
            return len(new_chunks)

        except Exception as e:
            print(f"⚠️  Error during incremental update: {e}")
            print("Falling back to full regeneration...")
            self._build_embeddings()
            return len(self.chunks)

    @classmethod
    def load_embeddings(cls, chunks: List[Dict], embedding_path: str = 'rag_system/transformer_embeddings.npz'):
        """Load pre-computed embeddings (uses incremental update automatically)"""
        if not Path(embedding_path).exists():
            raise FileNotFoundError(f"Embeddings not found at {embedding_path}")

        data = np.load(embedding_path, allow_pickle=True)
        model_name = str(data['model_name'])

        print(f"Loading embeddings from {embedding_path}")
        # use_incremental=True will auto-detect changes and only update what's needed
        retriever = cls(chunks, model_name=model_name, use_incremental=True)

        return retriever


def upgrade_rag_system():
    """
    Upgrade existing TF-IDF RAG to transformer-based semantic search
    Uses incremental updates - only regenerates embeddings for changed chunks
    """
    print("="*70)
    print("LAT5150DRVMIL RAG System - Transformer Upgrade")
    print("="*70)
    print("\nUpgrading from TF-IDF (51.8%) to Transformers (target: 88%+)")
    print("Using incremental updates (10x faster on re-runs)")
    print()

    # Load existing chunks
    chunks_file = 'rag_system/processed_docs.json'
    if not Path(chunks_file).exists():
        print(f"❌ {chunks_file} not found!")
        print("Run document_processor.py first to create the index.")
        return

    with open(chunks_file, 'r') as f:
        data = json.load(f)

    chunks = data['chunks']
    print(f"Loaded {len(chunks)} chunks from {chunks_file}")

    # Build transformer retriever (incremental by default)
    retriever = TransformerRetriever(chunks, use_incremental=True)

    # Save embeddings for fast loading
    retriever.save_embeddings()

    # Test with sample queries
    print("\n" + "="*70)
    print("Testing Upgraded System")
    print("="*70)

    test_queries = [
        "What is DSMIL activation?",
        "How to enable NPU modules?",
        "APT41 security hardening",
        "Kernel build process"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        results = retriever.search(query, top_k=3)

        for i, (chunk, score) in enumerate(results, 1):
            print(f"  Result {i} (Score: {score:.3f}):")
            print(f"    File: {chunk['metadata']['filepath']}")
            print(f"    Preview: {chunk['text'][:100]}...")

    print("\n" + "="*70)
    print("✓ Upgrade Complete!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Run: python3 rag_system/test_transformer_rag.py")
    print("  2. Compare accuracy with TF-IDF baseline")
    print("  3. Use transformer_query.py for semantic search")
    print()


if __name__ == '__main__':
    upgrade_rag_system()
