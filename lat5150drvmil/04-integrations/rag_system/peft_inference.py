#!/usr/bin/env python3
"""
PEFT Inference Script
Test fine-tuned embedding model performance
"""

import json
import time
from pathlib import Path
from typing import List, Tuple, Dict

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
except ImportError:
    print("❌ sentence-transformers not installed!")
    print("Install with: pip install sentence-transformers")
    import sys
    sys.exit(1)


class PEFTInference:
    """Inference using fine-tuned PEFT model"""

    def __init__(
        self,
        model_path='rag_system/peft_model',
        use_onnx=False
    ):
        """
        Load fine-tuned model

        Args:
            model_path: Path to fine-tuned model
            use_onnx: Use ONNX-optimized version if available
        """
        self.model_path = model_path

        # Check if model exists
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Fine-tuned model not found: {model_path}\n"
                "Run peft_finetune.py first to create the model."
            )

        # Check ONNX version
        onnx_path = Path(model_path) / "onnx"
        if use_onnx and onnx_path.exists():
            print(f"Loading ONNX-optimized model from {onnx_path}")
            try:
                from optimum.onnxruntime import ORTModelForFeatureExtraction
                from transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(onnx_path)
                self.model = ORTModelForFeatureExtraction.from_pretrained(onnx_path)
                self.is_onnx = True
                print("✓ ONNX model loaded (optimized inference)")
            except ImportError:
                print("⚠️  Optimum not available, falling back to standard model")
                self.model = SentenceTransformer(model_path)
                self.is_onnx = False
        else:
            print(f"Loading fine-tuned model from {model_path}")
            self.model = SentenceTransformer(model_path)
            self.is_onnx = False

        print(f"✓ Fine-tuned model loaded")
        print()

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings"""
        if self.is_onnx:
            # ONNX encoding
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            outputs = self.model(**inputs)
            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
            return embeddings
        else:
            # Standard encoding
            return self.model.encode(texts, convert_to_numpy=True)

    def compare_with_baseline(self, test_queries: List[str]):
        """
        Compare fine-tuned model with baseline

        Args:
            test_queries: List of test queries
        """
        print("=" * 70)
        print("Comparing Fine-Tuned vs Baseline")
        print("=" * 70)
        print()

        # Load baseline model
        print("Loading baseline model (BAAI/bge-base-en-v1.5)...")
        baseline = SentenceTransformer('BAAI/bge-base-en-v1.5')

        # Load document chunks
        with open('rag_system/processed_docs.json', 'r') as f:
            data = json.load(f)
        chunks = data['chunks']
        chunk_texts = [c['text'] for c in chunks]

        # Encode chunks with both models
        print("Encoding documents with baseline model...")
        start = time.time()
        baseline_embeddings = baseline.encode(chunk_texts, show_progress_bar=True)
        baseline_time = time.time() - start

        print()
        print("Encoding documents with fine-tuned model...")
        start = time.time()
        finetuned_embeddings = self.encode(chunk_texts)
        finetuned_time = time.time() - start

        print()
        print(f"Baseline encoding time: {baseline_time:.2f}s")
        print(f"Fine-tuned encoding time: {finetuned_time:.2f}s")
        print(f"Speedup: {baseline_time / finetuned_time:.2f}x")
        print()

        # Test queries
        print("=" * 70)
        print("Query Comparison")
        print("=" * 70)
        print()

        for query in test_queries:
            print(f"Query: {query}")
            print("-" * 70)

            # Baseline results
            query_emb_baseline = baseline.encode(query)
            similarities_baseline = np.dot(baseline_embeddings, query_emb_baseline) / (
                np.linalg.norm(baseline_embeddings, axis=1) * np.linalg.norm(query_emb_baseline)
            )
            top_idx_baseline = np.argsort(similarities_baseline)[::-1][:3]

            # Fine-tuned results
            query_emb_finetuned = self.encode([query])[0]
            similarities_finetuned = np.dot(finetuned_embeddings, query_emb_finetuned) / (
                np.linalg.norm(finetuned_embeddings, axis=1) * np.linalg.norm(query_emb_finetuned)
            )
            top_idx_finetuned = np.argsort(similarities_finetuned)[::-1][:3]

            # Compare
            print()
            print("BASELINE:")
            for i, idx in enumerate(top_idx_baseline, 1):
                print(f"  {i}. Score: {similarities_baseline[idx]:.3f}")
                print(f"     File: {chunks[idx]['metadata']['filepath']}")
                print(f"     Text: {chunks[idx]['text'][:100]}...")
                print()

            print("FINE-TUNED:")
            for i, idx in enumerate(top_idx_finetuned, 1):
                print(f"  {i}. Score: {similarities_finetuned[idx]:.3f}")
                print(f"     File: {chunks[idx]['metadata']['filepath']}")
                print(f"     Text: {chunks[idx]['text'][:100]}...")
                print()

            print("=" * 70)
            print()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='PEFT model inference testing')
    parser.add_argument(
        '--model-path',
        type=str,
        default='rag_system/peft_model',
        help='Path to fine-tuned model'
    )
    parser.add_argument(
        '--use-onnx',
        action='store_true',
        help='Use ONNX-optimized model'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare with baseline model'
    )

    args = parser.parse_args()

    # Initialize
    inference = PEFTInference(
        model_path=args.model_path,
        use_onnx=args.use_onnx
    )

    # Test queries
    test_queries = [
        "What is DSMIL activation?",
        "How to enable NPU modules?",
        "APT41 security hardening",
        "Kernel build process"
    ]

    if args.compare:
        inference.compare_with_baseline(test_queries)
    else:
        print("Fine-tuned model loaded successfully!")
        print()
        print("Usage:")
        print("  --compare: Compare with baseline model")
        print()


if __name__ == '__main__':
    main()
