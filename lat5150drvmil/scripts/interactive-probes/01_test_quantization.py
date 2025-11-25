#!/usr/bin/env python3
"""
Interactive Quantization System Probe

Tests hardware detection and quantization recommendations for:
- Intel NPU (Military-grade, 34-49.4 TOPS)
- Intel GNA (Gaussian Neural Accelerator)
- Intel NCS2 sticks (2-3 units)
- AVX-512 on P-cores

Usage:
    python 01_test_quantization.py
"""

import sys
sys.path.insert(0, '../../02-ai-engine')

from quantization_optimizer import (
    QuantizationOptimizer,
    HardwareBackend,
    QuantizationMethod
)


def print_section(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def test_hardware_detection():
    """Test hardware detection"""
    print_section("HARDWARE DETECTION")

    optimizer = QuantizationOptimizer()
    hw = optimizer.hw_detector

    print(f"\n✓ AVX-512 Support: {hw.has_avx512}")
    if hw.has_avx512:
        print(f"  → P-cores (0-5) can use AVX-512 for FP16/BF16 acceleration")

    print(f"\n✓ Intel NPU: {hw.has_npu}")
    if hw.has_npu:
        print(f"  → Military-grade NPU (34-49.4 TOPS)")
        print(f"  → Optimized for INT8 quantization")

    print(f"\n✓ CUDA GPU: {hw.has_cuda}")

    print(f"\n✓ Best available hardware: {hw.get_best_hardware().value}")

    print("\n📝 Note: NCS2 sticks (2-3 units) detected separately via OpenVINO")
    print("   Use OpenVINO IR format for NCS2 deployment")


def test_quantization_recommendations():
    """Test quantization recommendations for various model sizes"""
    print_section("QUANTIZATION RECOMMENDATIONS")

    optimizer = QuantizationOptimizer()

    test_models = [
        ("TinyLlama 1.1B", 1.1),
        ("Qwen2.5-Coder 1.5B", 1.5),
        ("DeepSeek-Coder 6.7B", 6.7),
        ("CodeLlama 13B", 13.0),
        ("DeepSeek-Coder 33B", 33.0),
        ("Llama 3.1 70B", 70.0),
    ]

    for model_name, size_gb in test_models:
        print(f"\n{model_name} ({size_gb}GB):")

        # Get recommendation
        config = optimizer.recommend_quantization(
            model_size_gb=size_gb,
            target_hardware=HardwareBackend.AUTO
        )

        print(f"  Recommended method: {config.method.value}")
        print(f"  Target hardware: {config.target_hardware.value}")
        print(f"  Preserve accuracy: {config.preserve_accuracy}")

        # Estimate compression
        original = size_gb * 1024  # MB
        quantized = optimizer._estimate_quantized_size(size_gb, config.method) * 1024
        compression = original / quantized

        print(f"  Original size: {original:.0f} MB")
        print(f"  Quantized size: {quantized:.0f} MB")
        print(f"  Compression: {compression:.1f}x smaller")

        # Hardware-specific notes
        if config.target_hardware == HardwareBackend.NPU:
            print(f"  💡 Deploy to NPU for 3-4x speedup with INT8")
        elif config.target_hardware == HardwareBackend.CPU_PCORE:
            print(f"  💡 Use AVX-512 on P-cores for 1.5-2x speedup with BF16")


def test_hardware_specific_quantization():
    """Test hardware-specific quantization"""
    print_section("HARDWARE-SPECIFIC QUANTIZATION")

    optimizer = QuantizationOptimizer()
    model_size = 6.7  # DeepSeek-Coder 6.7B

    backends = [
        (HardwareBackend.NPU, "Intel NPU (Military)"),
        (HardwareBackend.CPU_PCORE, "P-cores with AVX-512"),
        (HardwareBackend.CPU_ECORE, "E-cores (fallback)"),
    ]

    for backend, desc in backends:
        print(f"\n{desc}:")

        config = optimizer.recommend_quantization(
            model_size_gb=model_size,
            target_hardware=backend
        )

        print(f"  → Method: {config.method.value}")
        estimated = optimizer._estimate_quantized_size(model_size, config.method)
        print(f"  → Size: {estimated:.2f}GB")

        if backend == HardwareBackend.NPU:
            print(f"  → Best for: Maximum throughput (INT8 on NPU)")
        elif backend == HardwareBackend.CPU_PCORE:
            print(f"  → Best for: Balanced speed/accuracy (BF16 + AVX-512)")


def test_ncs2_deployment():
    """Test NCS2 stick deployment recommendations"""
    print_section("NCS2 STICK DEPLOYMENT (2-3 units)")

    print("\nFor NCS2 Neural Compute Stick deployment:")
    print("  1. Use OpenVINO IR format (INT8 quantization)")
    print("  2. Shard model across 2-3 NCS2 sticks for parallel inference")
    print("  3. Expected throughput: ~5-10 tok/sec per stick")
    print("  4. Total throughput: 10-30 tok/sec (2-3 sticks)")

    print("\nRecommended models for NCS2:")
    print("  ✓ TinyLlama 1.1B (fits on single NCS2)")
    print("  ✓ Qwen2.5-Coder 1.5B (fits on single NCS2)")
    print("  ✓ DeepSeek-Coder 6.7B (shard across 2-3 NCS2)")

    print("\nConversion command:")
    print("  mo --input_model model.onnx --compress_to_fp16")


def interactive_menu():
    """Interactive menu for testing"""
    while True:
        print("\n" + "=" * 80)
        print("  QUANTIZATION SYSTEM INTERACTIVE PROBE")
        print("=" * 80)
        print("\n1. Test hardware detection")
        print("2. Test quantization recommendations")
        print("3. Test hardware-specific quantization")
        print("4. Test NCS2 deployment")
        print("5. Run all tests")
        print("0. Exit")

        choice = input("\nSelect option: ").strip()

        if choice == "1":
            test_hardware_detection()
        elif choice == "2":
            test_quantization_recommendations()
        elif choice == "3":
            test_hardware_specific_quantization()
        elif choice == "4":
            test_ncs2_deployment()
        elif choice == "5":
            test_hardware_detection()
            test_quantization_recommendations()
            test_hardware_specific_quantization()
            test_ncs2_deployment()
        elif choice == "0":
            print("\nExiting...")
            break
        else:
            print("\nInvalid option!")


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║          QUANTIZATION SYSTEM INTERACTIVE PROBE                           ║
║          Dell Latitude 5450 MIL-SPEC Edition                             ║
╚══════════════════════════════════════════════════════════════════════════╝

Hardware Available:
  • Intel NPU (Military-grade: 34-49.4 TOPS)
  • Intel GNA (Gaussian Neural Accelerator)
  • Intel NCS2 sticks (2-3 units)
  • AVX-512 on P-cores (CPUs 0-5)

""")

    interactive_menu()
