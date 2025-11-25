#!/usr/bin/env python3
"""
Test Corrected Hardware Specifications
========================================
Verify that all specification corrections are working correctly.
"""

import sys
sys.path.insert(0, '/home/user/LAT5150DRVMIL/02-ai-engine')

from hardware_profile import get_hardware_profile
from dynamic_allocator import get_allocator, ModelSpec, QuantizationType
from whiterabbit_model_manager import WhiteRabbitModelManager

def test_hardware_profile():
    """Test hardware profile has correct values."""
    print("=" * 70)
    print("TEST 1: Hardware Profile Verification")
    print("=" * 70)

    profile = get_hardware_profile()

    # Expected values
    expected = {
        "system_ram_gb": 62.0,
        "usable_ram_gb": 56.0,
        "arc_gpu_tops_int8": 40.0,
        "npu_tops_optimized": 26.4,
        "ncs2_device_count": 1,
        "ncs2_inference_memory_mb": 512.0,
        "ncs2_storage_per_device_gb": 16.0,
        "ncs2_total_tops": 10.0,
        "total_system_tops": 76.4,
    }

    errors = []

    for key, expected_value in expected.items():
        actual_value = getattr(profile, key)
        if actual_value != expected_value:
            errors.append(f"  ✗ {key}: expected {expected_value}, got {actual_value}")
        else:
            print(f"  ✓ {key}: {actual_value}")

    if errors:
        print("\nERRORS FOUND:")
        for error in errors:
            print(error)
        return False
    else:
        print("\n✅ All hardware specs correct!")
        return True


def test_allocation_planning():
    """Test allocation planning with corrected specs."""
    print("\n" + "=" * 70)
    print("TEST 2: Allocation Planning with Corrected Specs")
    print("=" * 70)

    allocator = get_allocator()

    # Test WhiteRabbitNeo-33B-v1 allocation
    print("\nTesting WhiteRabbitNeo-33B-v1 (INT4)...")
    spec_33b = ModelSpec(
        name="WhiteRabbitNeo-33B-v1",
        params_billions=33.0,
        context_length=4096,
        num_layers=60
    )

    plan_33b = allocator.create_allocation_plan(
        model_spec=spec_33b,
        quantization=QuantizationType.INT4,
        enable_swap=True
    )

    # Test WhiteRabbitNeo-70B allocation
    print("\nTesting Llama-3.1-WhiteRabbitNeo-2-70B (INT4)...")
    spec_70b = ModelSpec(
        name="Llama-3.1-WhiteRabbitNeo-2-70B",
        params_billions=70.0,
        context_length=8192,
        num_layers=80
    )

    plan_70b = allocator.create_allocation_plan(
        model_spec=spec_70b,
        quantization=QuantizationType.INT4,
        enable_swap=True
    )

    # Verify plans are feasible
    success = True
    if not plan_33b.is_feasible:
        print("✗ 33B model allocation not feasible!")
        success = False
    else:
        print("✅ 33B model allocation is feasible")

    if not plan_70b.is_feasible:
        print("✗ 70B model allocation not feasible!")
        success = False
    else:
        print("✅ 70B model allocation is feasible")

    return success


def test_model_manager():
    """Test model manager integration."""
    print("\n" + "=" * 70)
    print("TEST 3: Model Manager Integration")
    print("=" * 70)

    manager = WhiteRabbitModelManager()

    # List models
    models = manager.list_models()
    print(f"\nAvailable models: {models}")

    # Plan allocation for both models
    for model_name in models:
        print(f"\nPlanning allocation for {model_name}...")
        plan = manager.plan_allocation(
            model_name=model_name,
            quantization=QuantizationType.INT4,
            enable_swap=True
        )

        if plan is None:
            print(f"✗ Failed to create plan for {model_name}")
            return False

        print(f"✅ Plan created for {model_name}")
        print(f"   Total memory: {plan.total_memory_required_gb:.1f} GB")
        print(f"   GPU layers: {len(plan.gpu_layers)}")
        print(f"   Feasible: {plan.is_feasible}")

    return True


def test_device_capabilities():
    """Test device capability detection."""
    print("\n" + "=" * 70)
    print("TEST 4: Device Capability Detection")
    print("=" * 70)

    allocator = get_allocator()

    expected_devices = {
        "GPU_ARC": {"memory_gb": 56.0, "compute_tops": 40.0},
        "NPU": {"memory_gb": 0.125, "compute_tops": 26.4},  # 128MB = 0.125GB
        "NCS2": {"memory_gb": 0.5, "compute_tops": 10.0},  # 512MB = 0.5GB
    }

    from dynamic_allocator import DeviceType

    errors = []
    for device_name, expected in expected_devices.items():
        device_type = DeviceType[device_name]
        device = allocator.devices[device_type]

        print(f"\n{device_name}:")
        print(f"  Memory: {device.memory_gb:.3f} GB")
        print(f"  Compute: {device.compute_tops:.1f} TOPS")

        # Check values (with tolerance for floating point)
        if abs(device.memory_gb - expected["memory_gb"]) > 0.01:
            errors.append(f"  ✗ {device_name} memory: expected {expected['memory_gb']:.3f} GB, got {device.memory_gb:.3f} GB")

        if abs(device.compute_tops - expected["compute_tops"]) > 0.1:
            errors.append(f"  ✗ {device_name} compute: expected {expected['compute_tops']:.1f} TOPS, got {device.compute_tops:.1f} TOPS")

    if errors:
        print("\nERRORS FOUND:")
        for error in errors:
            print(error)
        return False
    else:
        print("\n✅ All device capabilities correct!")
        return True


def main():
    """Run all tests."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║            WhiteRabbitNeo Specification Correction Test              ║
║                                                                      ║
║  Verifying corrected hardware specifications                        ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

    tests = [
        test_hardware_profile,
        test_device_capabilities,
        test_allocation_planning,
        test_model_manager,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n❌ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(results)
    total = len(results)

    print(f"\nPassed: {passed}/{total}")

    if all(results):
        print("\n✅ ALL TESTS PASSED!")
        print("\nCorrected specifications:")
        print("  • System RAM: 62GB (56GB usable)")
        print("  • Arc GPU: 40 TOPS INT8")
        print("  • NPU: 26.4 TOPS (military mode)")
        print("  • NCS2: 1 device, 10 TOPS, 512MB inference memory")
        print("  • Total Compute: 76.4 TOPS")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
