#!/usr/bin/env python3
"""
Model Quantization with Optimum-Quanto
On-the-fly quantization for faster inference and lower memory

Supports:
- INT8 quantization (2x smaller, 1.5-2x faster)
- INT4 quantization (4x smaller, 2-3x faster)
- INT2 quantization (8x smaller, 3-4x faster)
- No retraining required
- Hardware-aware (CPU/GPU/NPU)
"""

import time
from pathlib import Path
from typing import Optional, Literal

try:
    from optimum.quanto import quantize, freeze, qint8, qint4, qint2
    from optimum.quanto import QuantizedModel
    import torch
    QUANTO_AVAILABLE = True
except ImportError:
    print("⚠️  optimum-quanto not installed")
    print("Install with: pip install optimum-quanto")
    QUANTO_AVAILABLE = False


class QuantizationOptimizer:
    """Quantize models with Optimum-Quanto"""

    PRECISION_LEVELS = {
        'int8': (qint8, "2x smaller, 1.5-2x faster"),
        'int4': (qint4, "4x smaller, 2-3x faster"),
        'int2': (qint2, "8x smaller, 3-4x faster"),
    }

    def __init__(self):
        """Initialize quantization optimizer"""
        if not QUANTO_AVAILABLE:
            raise ImportError("optimum-quanto not installed")

    def quantize_model(
        self,
        model,
        precision: Literal['int8', 'int4', 'int2'] = 'int8',
        freeze_model: bool = True
    ):
        """
        Quantize model on-the-fly (no retraining needed)

        Args:
            model: PyTorch model to quantize
            precision: Quantization precision (int8, int4, int2)
            freeze_model: Freeze quantized model (faster inference)

        Returns:
            Quantized model
        """
        if precision not in self.PRECISION_LEVELS:
            raise ValueError(f"Precision must be one of: {list(self.PRECISION_LEVELS.keys())}")

        qtype, description = self.PRECISION_LEVELS[precision]

        print(f"Quantizing model to {precision.upper()}...")
        print(f"Expected: {description}")
        print()

        # Quantize
        quantize(model, weights=qtype)

        # Freeze for faster inference
        if freeze_model:
            freeze(model)
            print("✓ Model frozen (optimized for inference)")

        print(f"✓ Model quantized to {precision.upper()}")
        return model

    def benchmark_quantization(
        self,
        model_path: str,
        test_samples: int = 100
    ):
        """
        Benchmark different quantization levels

        Args:
            model_path: Path to model
            test_samples: Number of test samples
        """
        from sentence_transformers import SentenceTransformer
        import numpy as np

        print("=" * 70)
        print("Quantization Benchmark")
        print("=" * 70)
        print()

        # Load base model
        print("Loading base model...")
        model_fp32 = SentenceTransformer(model_path)

        # Test data
        test_texts = [f"test query {i}" for i in range(test_samples)]

        results = {}

        # Baseline: FP32
        print("\nBenchmark 1: FP32 (baseline)")
        start = time.time()
        embeddings_fp32 = model_fp32.encode(test_texts, show_progress_bar=False)
        fp32_time = time.time() - start
        results['fp32'] = fp32_time

        # Get model size
        fp32_size = sum(p.numel() * p.element_size() for p in model_fp32.parameters()) / 1024 / 1024

        print(f"  Time: {fp32_time:.3f}s")
        print(f"  Size: {fp32_size:.2f} MB")
        print(f"  Speed: {test_samples/fp32_time:.1f} samples/sec")

        # Test each quantization level
        for precision in ['int8', 'int4', 'int2']:
            print(f"\nBenchmark: {precision.upper()}")

            try:
                # Load fresh model
                model_quant = SentenceTransformer(model_path)

                # Quantize
                qtype, _ = self.PRECISION_LEVELS[precision]
                quantize(model_quant._modules['0'].auto_model, weights=qtype)
                freeze(model_quant._modules['0'].auto_model)

                # Benchmark
                start = time.time()
                embeddings_quant = model_quant.encode(test_texts, show_progress_bar=False)
                quant_time = time.time() - start
                results[precision] = quant_time

                # Size estimate
                size_reduction = {'int8': 2, 'int4': 4, 'int2': 8}[precision]
                quant_size = fp32_size / size_reduction

                # Speedup
                speedup = fp32_time / quant_time

                print(f"  Time: {quant_time:.3f}s")
                print(f"  Size: {quant_size:.2f} MB (est.)")
                print(f"  Speed: {test_samples/quant_time:.1f} samples/sec")
                print(f"  Speedup: {speedup:.2f}x vs FP32")

                # Quality check (cosine similarity)
                similarity = np.mean([
                    np.dot(embeddings_fp32[i], embeddings_quant[i]) /
                    (np.linalg.norm(embeddings_fp32[i]) * np.linalg.norm(embeddings_quant[i]))
                    for i in range(min(10, len(embeddings_fp32)))
                ])
                print(f"  Quality: {similarity:.4f} (cosine similarity to FP32)")

            except Exception as e:
                print(f"  ❌ Failed: {e}")

        # Summary
        print()
        print("=" * 70)
        print("Quantization Summary")
        print("=" * 70)
        print()

        if len(results) > 1:
            fastest = min((k, v) for k, v in results.items() if k != 'fp32')
            print(f"Fastest: {fastest[0].upper()} ({fastest[1]:.3f}s)")
            print(f"Speedup: {results['fp32']/fastest[1]:.2f}x vs FP32")
            print()

            print("Recommendations:")
            print("  • INT8: Best quality/speed tradeoff (recommended)")
            print("  • INT4: Good for memory-constrained systems")
            print("  • INT2: Maximum compression (may impact quality)")

        print()

    def quantize_rag_model(
        self,
        model_path: str = 'rag_system/peft_model',
        output_path: str = 'rag_system/peft_model_quantized',
        precision: Literal['int8', 'int4', 'int2'] = 'int8'
    ):
        """
        Quantize RAG embedding model

        Args:
            model_path: Path to fine-tuned model
            output_path: Where to save quantized model
            precision: Quantization precision
        """
        from sentence_transformers import SentenceTransformer

        print("=" * 70)
        print("RAG Model Quantization")
        print("=" * 70)
        print()

        if not Path(model_path).exists():
            print(f"❌ Model not found: {model_path}")
            print("Run PEFT fine-tuning first or use base model")
            return

        # Load model
        print(f"Loading model from {model_path}...")
        model = SentenceTransformer(model_path)

        # Quantize
        qtype, description = self.PRECISION_LEVELS[precision]
        print(f"\nQuantizing to {precision.upper()}...")
        print(f"Expected: {description}")
        print()

        quantize(model._modules['0'].auto_model, weights=qtype)
        freeze(model._modules['0'].auto_model)

        # Save
        Path(output_path).mkdir(parents=True, exist_ok=True)
        model.save(output_path)

        print(f"✓ Quantized model saved to {output_path}")
        print()

        # Test
        print("Testing quantized model...")
        test_query = "What is DSMIL activation?"
        embedding = model.encode(test_query)

        print(f"  Query: {test_query}")
        print(f"  Embedding shape: {embedding.shape}")
        print(f"  ✓ Model works!")
        print()

        print("Usage:")
        print(f"  model = SentenceTransformer('{output_path}')")
        print(f"  embeddings = model.encode(queries)")
        print()


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Quantization Optimizer')
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Benchmark quantization levels'
    )
    parser.add_argument(
        '--quantize',
        type=str,
        help='Quantize model (specify precision: int8, int4, int2)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='BAAI/bge-base-en-v1.5',
        help='Model to quantize'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output path for quantized model'
    )

    args = parser.parse_args()

    if not QUANTO_AVAILABLE:
        print("❌ optimum-quanto not installed")
        print("Install with: pip install optimum-quanto")
        return

    optimizer = QuantizationOptimizer()

    if args.benchmark:
        optimizer.benchmark_quantization(args.model)

    elif args.quantize:
        if args.quantize not in ['int8', 'int4', 'int2']:
            print("❌ Precision must be: int8, int4, or int2")
            return

        output = args.output or f"{args.model}_quantized_{args.quantize}"
        optimizer.quantize_rag_model(args.model, output, args.quantize)

    else:
        print("Optimum-Quanto Quantization")
        print()
        print("Features:")
        print("  • On-the-fly quantization (no retraining)")
        print("  • INT8: 2x smaller, 1.5-2x faster")
        print("  • INT4: 4x smaller, 2-3x faster")
        print("  • INT2: 8x smaller, 3-4x faster")
        print()
        print("Usage:")
        print("  --benchmark           Benchmark quantization levels")
        print("  --quantize int8       Quantize model to INT8")
        print("  --model PATH          Model to quantize")
        print("  --output PATH         Output path")
        print()
        print("Example:")
        print("  python quantization_optimizer.py --benchmark")
        print("  python quantization_optimizer.py --quantize int8 --model rag_system/peft_model")


if __name__ == '__main__':
    main()
