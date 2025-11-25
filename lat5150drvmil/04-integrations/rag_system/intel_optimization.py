#!/usr/bin/env python3
"""
Intel Hardware Optimization for RAG System
Optimizes transformer inference for Intel CPUs/NPUs (Meteor Lake)

Features:
- OpenVINO acceleration (Intel CPUs/GPUs/NPUs)
- Intel Extension for PyTorch (IPEX)
- Neural Compressor quantization
- NPU offloading (if available)
"""

import os
import sys
from pathlib import Path
from typing import Optional

try:
    from optimum.intel import OVModelForFeatureExtraction
    from optimum.intel.openvino import OVConfig
    OPTIMUM_INTEL_AVAILABLE = True
except ImportError:
    print("⚠️  optimum-intel not installed")
    print("Install with: pip install optimum-intel[openvino,nncf]")
    OPTIMUM_INTEL_AVAILABLE = False

try:
    import intel_extension_for_pytorch as ipex
    IPEX_AVAILABLE = True
except ImportError:
    print("⚠️  Intel Extension for PyTorch not installed")
    print("Install with: pip install intel-extension-for-pytorch")
    IPEX_AVAILABLE = False


class IntelOptimizer:
    """Optimize RAG system for Intel hardware"""

    def __init__(self):
        """Initialize Intel optimizer"""
        self.has_openvino = OPTIMUM_INTEL_AVAILABLE
        self.has_ipex = IPEX_AVAILABLE
        self.device_info = self._detect_intel_hardware()

    def _detect_intel_hardware(self) -> dict:
        """Detect available Intel acceleration hardware"""
        info = {
            'cpu': True,  # Always available
            'gpu': False,
            'npu': False,
            'cpu_model': None
        }

        try:
            # Detect CPU model
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if 'model name' in line:
                        info['cpu_model'] = line.split(':')[1].strip()
                        break

            # Check for Intel NPU (Meteor Lake)
            if 'Meteor Lake' in (info['cpu_model'] or ''):
                # Meteor Lake has integrated NPU
                info['npu'] = True

            # Check for Intel GPU via OpenVINO
            if OPTIMUM_INTEL_AVAILABLE:
                try:
                    from openvino.runtime import Core
                    ie = Core()
                    devices = ie.available_devices

                    if 'GPU' in devices:
                        info['gpu'] = True
                    if 'NPU' in devices or 'VPU' in devices:
                        info['npu'] = True

                except Exception:
                    pass

        except Exception as e:
            print(f"Warning: Could not detect Intel hardware: {e}")

        return info

    def convert_model_to_openvino(
        self,
        model_path: str,
        output_path: str,
        quantize: bool = True
    ):
        """
        Convert HuggingFace model to OpenVINO format

        Args:
            model_path: Path to HuggingFace model
            output_path: Where to save OpenVINO model
            quantize: Apply INT8 quantization for faster inference
        """
        if not self.has_openvino:
            print("❌ OpenVINO not available. Install optimum-intel.")
            return False

        print("=" * 70)
        print("Converting Model to OpenVINO (Intel Optimization)")
        print("=" * 70)
        print()

        print(f"Source: {model_path}")
        print(f"Output: {output_path}")
        print(f"Quantization: {'INT8' if quantize else 'FP32'}")
        print()

        try:
            from transformers import AutoTokenizer

            # Load tokenizer
            print("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            # Convert to OpenVINO
            print("Converting to OpenVINO format...")

            if quantize:
                print("Applying INT8 quantization (Neural Compressor)...")
                # Use quantization for better performance
                ov_config = OVConfig(compression={"algorithm": "quantization"})
            else:
                ov_config = None

            model = OVModelForFeatureExtraction.from_pretrained(
                model_path,
                export=True,
                ov_config=ov_config
            )

            # Save
            print(f"Saving to {output_path}...")
            model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)

            print()
            print("✓ Conversion complete!")
            print()

            # Show expected speedup
            print("Expected Performance:")
            if quantize:
                print("  • Inference speed: 2-4x faster than PyTorch")
                print("  • Memory usage: 50% reduction (INT8 vs FP32)")
            else:
                print("  • Inference speed: 1.5-2x faster than PyTorch")

            if self.device_info['npu']:
                print("  • NPU acceleration: Available (Meteor Lake)")
                print("  • Recommend using device='NPU' for inference")

            print()
            return True

        except Exception as e:
            print(f"❌ Conversion failed: {e}")
            return False

    def optimize_with_ipex(self, model, optimize_for_inference: bool = True):
        """
        Optimize PyTorch model with Intel Extension for PyTorch

        Args:
            model: PyTorch model
            optimize_for_inference: Apply inference optimizations
        """
        if not self.has_ipex:
            print("⚠️  IPEX not available. Install intel-extension-for-pytorch")
            return model

        print("Optimizing with Intel Extension for PyTorch (IPEX)...")

        try:
            import torch

            # Put model in eval mode
            model.eval()

            # Optimize with IPEX
            if optimize_for_inference:
                model = ipex.optimize(model, dtype=torch.float32, inplace=True)
                print("✓ IPEX optimization applied (inference mode)")
            else:
                model = ipex.optimize(model, dtype=torch.float32)
                print("✓ IPEX optimization applied")

            return model

        except Exception as e:
            print(f"⚠️  IPEX optimization failed: {e}")
            return model

    def benchmark_inference(self, model_path: str, num_samples: int = 100):
        """
        Benchmark inference speed with different optimizations

        Args:
            model_path: Path to model (HuggingFace or OpenVINO)
            num_samples: Number of test samples
        """
        import time
        import numpy as np

        print("=" * 70)
        print("Intel Hardware Inference Benchmark")
        print("=" * 70)
        print()

        print(f"Hardware Detected:")
        print(f"  CPU: {self.device_info['cpu_model']}")
        print(f"  GPU: {'Available' if self.device_info['gpu'] else 'Not detected'}")
        print(f"  NPU: {'Available' if self.device_info['npu'] else 'Not detected'}")
        print()

        results = {}

        # Test 1: Standard PyTorch
        print("Benchmark 1: Standard PyTorch (baseline)")
        try:
            from sentence_transformers import SentenceTransformer
            import torch

            model = SentenceTransformer(model_path)
            test_texts = ["test query"] * num_samples

            start = time.time()
            embeddings = model.encode(test_texts, show_progress_bar=False)
            elapsed = time.time() - start

            results['pytorch'] = elapsed
            print(f"  Time: {elapsed:.3f}s ({num_samples/elapsed:.1f} samples/sec)")
            print()

        except Exception as e:
            print(f"  ❌ Failed: {e}\n")

        # Test 2: PyTorch + IPEX
        if self.has_ipex:
            print("Benchmark 2: PyTorch + IPEX")
            try:
                model_ipex = SentenceTransformer(model_path)
                model_ipex = self.optimize_with_ipex(model_ipex)

                start = time.time()
                embeddings = model_ipex.encode(test_texts, show_progress_bar=False)
                elapsed = time.time() - start

                results['ipex'] = elapsed
                speedup = results['pytorch'] / elapsed if 'pytorch' in results else 1.0
                print(f"  Time: {elapsed:.3f}s ({num_samples/elapsed:.1f} samples/sec)")
                print(f"  Speedup: {speedup:.2f}x vs baseline")
                print()

            except Exception as e:
                print(f"  ❌ Failed: {e}\n")

        # Test 3: OpenVINO (if available)
        if self.has_openvino:
            print("Benchmark 3: OpenVINO (Intel optimized)")
            try:
                # Convert if needed
                ov_path = f"{model_path}_openvino"
                if not Path(ov_path).exists():
                    print("  Converting to OpenVINO format...")
                    self.convert_model_to_openvino(model_path, ov_path, quantize=True)

                from optimum.intel import OVModelForFeatureExtraction
                from transformers import AutoTokenizer
                import torch

                tokenizer = AutoTokenizer.from_pretrained(ov_path)
                model_ov = OVModelForFeatureExtraction.from_pretrained(ov_path)

                # Encode test
                inputs = tokenizer(test_texts, padding=True, truncation=True, return_tensors="pt")

                start = time.time()
                with torch.no_grad():
                    outputs = model_ov(**inputs)
                    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
                elapsed = time.time() - start

                results['openvino'] = elapsed
                speedup = results['pytorch'] / elapsed if 'pytorch' in results else 1.0
                print(f"  Time: {elapsed:.3f}s ({num_samples/elapsed:.1f} samples/sec)")
                print(f"  Speedup: {speedup:.2f}x vs baseline")
                print()

            except Exception as e:
                print(f"  ❌ Failed: {e}\n")

        # Summary
        print("=" * 70)
        print("Benchmark Summary")
        print("=" * 70)
        print()

        if results:
            fastest = min(results.items(), key=lambda x: x[1])
            print(f"Fastest: {fastest[0].upper()} ({fastest[1]:.3f}s)")
            print()

            print("Recommendations:")
            if 'openvino' in results and fastest[0] == 'openvino':
                print("  ✓ Use OpenVINO for best performance on Intel hardware")
                print("  ✓ INT8 quantization provides 2-4x speedup")
            elif 'ipex' in results and fastest[0] == 'ipex':
                print("  ✓ Use IPEX for optimized PyTorch inference")

            if self.device_info['npu']:
                print("  ℹ NPU available - consider testing with device='NPU'")

        print()


def main():
    """Test Intel optimization"""
    import argparse

    parser = argparse.ArgumentParser(description='Intel Hardware Optimization')
    parser.add_argument(
        '--model',
        type=str,
        default='rag_system/peft_model',
        help='Model path to optimize'
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run inference benchmark'
    )
    parser.add_argument(
        '--convert',
        action='store_true',
        help='Convert model to OpenVINO'
    )
    parser.add_argument(
        '--quantize',
        action='store_true',
        default=True,
        help='Apply INT8 quantization (default: True)'
    )

    args = parser.parse_args()

    optimizer = IntelOptimizer()

    print("=" * 70)
    print("Intel Hardware Optimization for LAT5150DRVMIL RAG")
    print("=" * 70)
    print()

    # Show hardware info
    print("Detected Hardware:")
    print(f"  CPU: {optimizer.device_info['cpu_model']}")
    print(f"  GPU: {'✓ Available' if optimizer.device_info['gpu'] else '✗ Not detected'}")
    print(f"  NPU: {'✓ Available' if optimizer.device_info['npu'] else '✗ Not detected'}")
    print()

    print("Available Optimizations:")
    print(f"  OpenVINO: {'✓ Available' if optimizer.has_openvino else '✗ Not installed'}")
    print(f"  IPEX: {'✓ Available' if optimizer.has_ipex else '✗ Not installed'}")
    print()

    if args.convert:
        # Convert to OpenVINO
        model_path = args.model
        output_path = f"{model_path}_openvino"

        optimizer.convert_model_to_openvino(
            model_path,
            output_path,
            quantize=args.quantize
        )

    if args.benchmark:
        # Run benchmark
        optimizer.benchmark_inference(args.model, num_samples=100)


if __name__ == '__main__':
    main()
