#!/usr/bin/env python3
"""
Safetensors Model Loader
Fast, safe, zero-copy model loading

Benefits over pickle:
- Zero-copy loading (instant deserialization)
- Lazy loading (load only needed weights)
- No arbitrary code execution (safer than pickle)
- Smaller file size
- Faster loading times (2-10x)
"""

import os
import json
from pathlib import Path
from typing import Dict, Optional

try:
    from safetensors import safe_open
    from safetensors.torch import save_file, load_file
    import torch
    SAFETENSORS_AVAILABLE = True
except ImportError:
    print("⚠️  safetensors not installed")
    print("Install with: pip install safetensors")
    SAFETENSORS_AVAILABLE = False


class SafeTensorLoader:
    """Load and save models using safetensors (zero-copy)"""

    @staticmethod
    def convert_pytorch_to_safetensors(
        pytorch_path: str,
        safetensors_path: str,
        metadata: Optional[Dict] = None
    ):
        """
        Convert PyTorch checkpoint to safetensors format

        Args:
            pytorch_path: Path to .pt or .pth file
            safetensors_path: Output path for .safetensors
            metadata: Optional metadata to include
        """
        if not SAFETENSORS_AVAILABLE:
            raise ImportError("safetensors not installed")

        print(f"Converting {pytorch_path} to safetensors...")

        # Load PyTorch checkpoint
        checkpoint = torch.load(pytorch_path, map_location='cpu')

        # Extract state dict (handle different checkpoint formats)
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Save as safetensors
        save_file(state_dict, safetensors_path, metadata=metadata)

        # Compare sizes
        pytorch_size = os.path.getsize(pytorch_path) / 1024 / 1024
        safetensors_size = os.path.getsize(safetensors_path) / 1024 / 1024

        print(f"✓ Converted successfully!")
        print(f"  PyTorch size: {pytorch_size:.2f} MB")
        print(f"  Safetensors size: {safetensors_size:.2f} MB")
        print(f"  Reduction: {(1 - safetensors_size/pytorch_size)*100:.1f}%")

        return safetensors_path

    @staticmethod
    def load_safetensors(path: str, device: str = 'cpu') -> Dict:
        """
        Load safetensors file (zero-copy, fast)

        Args:
            path: Path to .safetensors file
            device: Device to load tensors on

        Returns:
            State dict
        """
        if not SAFETENSORS_AVAILABLE:
            raise ImportError("safetensors not installed")

        return load_file(path, device=device)

    @staticmethod
    def load_safetensors_lazy(path: str) -> 'SafeTensorFile':
        """
        Open safetensors file for lazy loading

        This allows loading individual tensors without loading the entire file

        Args:
            path: Path to .safetensors file

        Returns:
            SafeTensorFile object for lazy access
        """
        if not SAFETENSORS_AVAILABLE:
            raise ImportError("safetensors not installed")

        return safe_open(path, framework="pt", device="cpu")

    @staticmethod
    def inspect_safetensors(path: str):
        """
        Inspect safetensors file without loading weights

        Args:
            path: Path to .safetensors file
        """
        if not SAFETENSORS_AVAILABLE:
            raise ImportError("safetensors not installed")

        print(f"Inspecting {path}")
        print("=" * 70)

        with safe_open(path, framework="pt", device="cpu") as f:
            # Get metadata
            metadata = f.metadata()
            if metadata:
                print("\nMetadata:")
                for key, value in metadata.items():
                    print(f"  {key}: {value}")

            # List tensors
            print("\nTensors:")
            total_size = 0
            tensor_count = 0

            for key in f.keys():
                tensor = f.get_tensor(key)
                size_mb = tensor.nelement() * tensor.element_size() / 1024 / 1024
                total_size += size_mb
                tensor_count += 1

                print(f"  {key}")
                print(f"    Shape: {tuple(tensor.shape)}")
                print(f"    Dtype: {tensor.dtype}")
                print(f"    Size: {size_mb:.2f} MB")

            print()
            print(f"Total tensors: {tensor_count}")
            print(f"Total size: {total_size:.2f} MB")

        print("=" * 70)


def benchmark_loading_speed():
    """Benchmark safetensors vs PyTorch loading speed"""
    import time
    import tempfile
    import torch

    print("=" * 70)
    print("Safetensors vs PyTorch Loading Benchmark")
    print("=" * 70)
    print()

    # Create test model
    print("Creating test model (100MB)...")
    test_model = {
        f'layer_{i}': torch.randn(1000, 1000)
        for i in range(25)  # ~100MB
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        pytorch_path = os.path.join(tmpdir, 'model.pt')
        safetensors_path = os.path.join(tmpdir, 'model.safetensors')

        # Save PyTorch
        print("Saving PyTorch format...")
        torch.save(test_model, pytorch_path)

        # Save safetensors
        print("Saving safetensors format...")
        if SAFETENSORS_AVAILABLE:
            save_file(test_model, safetensors_path)
        else:
            print("Safetensors not available, skipping...")
            return

        # Benchmark PyTorch loading
        print("\nBenchmark 1: PyTorch loading (pickle)")
        times_pytorch = []
        for i in range(5):
            start = time.time()
            _ = torch.load(pytorch_path, map_location='cpu')
            elapsed = time.time() - start
            times_pytorch.append(elapsed)
            print(f"  Run {i+1}: {elapsed:.3f}s")

        avg_pytorch = sum(times_pytorch) / len(times_pytorch)
        print(f"  Average: {avg_pytorch:.3f}s")

        # Benchmark safetensors loading
        print("\nBenchmark 2: Safetensors loading (zero-copy)")
        times_safetensors = []
        for i in range(5):
            start = time.time()
            _ = load_file(safetensors_path, device='cpu')
            elapsed = time.time() - start
            times_safetensors.append(elapsed)
            print(f"  Run {i+1}: {elapsed:.3f}s")

        avg_safetensors = sum(times_safetensors) / len(times_safetensors)
        print(f"  Average: {avg_safetensors:.3f}s")

        # Summary
        print()
        print("=" * 70)
        print("Results")
        print("=" * 70)
        speedup = avg_pytorch / avg_safetensors
        print(f"Safetensors speedup: {speedup:.2f}x faster")

        pytorch_size = os.path.getsize(pytorch_path) / 1024 / 1024
        safetensors_size = os.path.getsize(safetensors_path) / 1024 / 1024
        print(f"File size reduction: {(1-safetensors_size/pytorch_size)*100:.1f}%")
        print()
        print("Benefits:")
        print(f"  ✓ {speedup:.1f}x faster loading")
        print(f"  ✓ {(1-safetensors_size/pytorch_size)*100:.0f}% smaller file")
        print("  ✓ Zero-copy deserialization")
        print("  ✓ Lazy loading support")
        print("  ✓ No arbitrary code execution (safer)")
        print()


def convert_rag_embeddings_to_safetensors():
    """Convert RAG system embeddings to safetensors format"""
    import numpy as np
    import torch

    embeddings_path = 'rag_system/transformer_embeddings.npz'
    safetensors_path = 'rag_system/transformer_embeddings.safetensors'

    if not Path(embeddings_path).exists():
        print(f"❌ {embeddings_path} not found")
        return

    if not SAFETENSORS_AVAILABLE:
        print("❌ safetensors not installed")
        return

    print("Converting RAG embeddings to safetensors format...")
    print()

    # Load numpy embeddings
    data = np.load(embeddings_path)
    embeddings = data['embeddings']
    model_name = str(data['model_name'])

    print(f"Loaded embeddings: shape {embeddings.shape}")

    # Convert to PyTorch tensors
    tensors = {
        'embeddings': torch.from_numpy(embeddings),
    }

    # Save as safetensors with metadata
    metadata = {
        'model_name': model_name,
        'shape': str(embeddings.shape),
        'format': 'safetensors',
        'source': 'LAT5150DRVMIL RAG system'
    }

    save_file(tensors, safetensors_path, metadata=metadata)

    # Compare sizes
    numpy_size = os.path.getsize(embeddings_path) / 1024 / 1024
    safetensors_size = os.path.getsize(safetensors_path) / 1024 / 1024

    print()
    print("✓ Conversion complete!")
    print(f"  NumPy size: {numpy_size:.2f} MB")
    print(f"  Safetensors size: {safetensors_size:.2f} MB")
    print(f"  Speedup: Zero-copy loading (instant)")
    print()

    # Test loading speed
    import time

    print("Loading benchmark:")

    # NumPy loading
    start = time.time()
    _ = np.load(embeddings_path)['embeddings']
    numpy_time = time.time() - start
    print(f"  NumPy: {numpy_time:.3f}s")

    # Safetensors loading
    start = time.time()
    _ = load_file(safetensors_path)['embeddings']
    safetensors_time = time.time() - start
    print(f"  Safetensors: {safetensors_time:.3f}s")

    speedup = numpy_time / safetensors_time
    print(f"  Speedup: {speedup:.2f}x faster")
    print()


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Safetensors Model Loader')
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run loading speed benchmark'
    )
    parser.add_argument(
        '--convert-embeddings',
        action='store_true',
        help='Convert RAG embeddings to safetensors'
    )
    parser.add_argument(
        '--convert',
        type=str,
        help='Convert PyTorch model to safetensors'
    )
    parser.add_argument(
        '--inspect',
        type=str,
        help='Inspect safetensors file'
    )

    args = parser.parse_args()

    if args.benchmark:
        benchmark_loading_speed()

    elif args.convert_embeddings:
        convert_rag_embeddings_to_safetensors()

    elif args.convert:
        output = args.convert.replace('.pt', '.safetensors').replace('.pth', '.safetensors')
        SafeTensorLoader.convert_pytorch_to_safetensors(args.convert, output)

    elif args.inspect:
        SafeTensorLoader.inspect_safetensors(args.inspect)

    else:
        print("Safetensors Model Loader")
        print()
        print("Usage:")
        print("  --benchmark              Run loading speed benchmark")
        print("  --convert-embeddings     Convert RAG embeddings to safetensors")
        print("  --convert MODEL.pt       Convert PyTorch model to safetensors")
        print("  --inspect MODEL.safetensors  Inspect safetensors file")


if __name__ == '__main__':
    main()
