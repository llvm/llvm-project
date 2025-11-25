#!/usr/bin/env python3
"""
Verify Specification Corrections
=================================
Check that all files have been corrected with accurate specifications.
"""

import re
from pathlib import Path


def check_file_for_patterns(filepath, correct_patterns, incorrect_patterns):
    """
    Check file contains correct patterns and does NOT contain incorrect patterns.

    Args:
        filepath: Path to file
        correct_patterns: List of (pattern, description) tuples that SHOULD be present
        incorrect_patterns: List of (pattern, description) tuples that SHOULD NOT be present

    Returns:
        (success, errors) tuple
    """
    try:
        content = Path(filepath).read_text()
    except Exception as e:
        return False, [f"Failed to read {filepath}: {e}"]

    errors = []

    # Check for correct patterns
    for pattern, description in correct_patterns:
        if not re.search(pattern, content):
            errors.append(f"  ✗ Missing: {description}")

    # Check for incorrect patterns
    for pattern, description in incorrect_patterns:
        if re.search(pattern, content):
            matches = re.findall(pattern, content)
            errors.append(f"  ✗ Found incorrect: {description} (found: {matches[:3]})")

    return len(errors) == 0, errors


def test_hardware_profile():
    """Test hardware_profile.py corrections."""
    print("=" * 70)
    print("TEST 1: hardware_profile.py")
    print("=" * 70)

    filepath = "/home/user/LAT5150DRVMIL/02-ai-engine/hardware_profile.py"

    correct_patterns = [
        (r"system_ram_gb:\s*float\s*=\s*62\.0", "System RAM: 62.0 GB"),
        (r"usable_ram_gb:\s*float\s*=\s*56\.0", "Usable RAM: 56.0 GB"),
        (r"arc_gpu_tops_int8:\s*float\s*=\s*40\.0", "Arc GPU: 40.0 TOPS"),
        (r"npu_tops_optimized:\s*float\s*=\s*26\.4", "NPU: 26.4 TOPS"),
        (r"ncs2_device_count:\s*int\s*=\s*1", "NCS2 count: 1 device"),
        (r"ncs2_inference_memory_mb:\s*float\s*=\s*512\.0", "NCS2 inference: 512 MB"),
        (r"total_system_tops:\s*float\s*=\s*76\.4", "Total TOPS: 76.4"),
        (r"128MB BAR0", "NPU memory description corrected"),
    ]

    incorrect_patterns = [
        (r"system_ram_gb:\s*float\s*=\s*64\.0", "Old incorrect RAM: 64 GB"),
        (r"arc_gpu_tops_int8:\s*float\s*=\s*100\.0", "Old incorrect GPU: 100 TOPS"),
        (r"npu_tops_optimized:\s*float\s*=\s*30\.0", "Old incorrect NPU: 30 TOPS"),
        (r"ncs2_device_count:\s*int\s*=\s*3", "Old incorrect NCS2 count: 3"),
        (r"total_system_tops:\s*float\s*=\s*160\.0", "Old incorrect total: 160 TOPS"),
        (r"total_system_tops:\s*float\s*=\s*260\.0", "Old incorrect total: 260 TOPS"),
    ]

    success, errors = check_file_for_patterns(filepath, correct_patterns, incorrect_patterns)

    if success:
        print("✅ All corrections verified!")
    else:
        print("❌ Errors found:")
        for error in errors:
            print(error)

    return success


def test_dynamic_allocator():
    """Test dynamic_allocator.py corrections."""
    print("\n" + "=" * 70)
    print("TEST 2: dynamic_allocator.py")
    print("=" * 70)

    filepath = "/home/user/LAT5150DRVMIL/02-ai-engine/dynamic_allocator.py"

    correct_patterns = [
        (r"CORRECTED Memory Architecture", "Header corrected"),
        (r"62GB", "62GB RAM mentioned"),
        (r"56GB usable", "56GB usable mentioned"),
        (r"40 TOPS INT8", "40 TOPS mentioned for Arc"),
        (r"26\.4 TOPS", "26.4 TOPS mentioned for NPU"),
        (r"512MB inference memory", "NCS2 inference memory mentioned"),
        (r"from hardware_profile import get_hardware_profile", "Imports hardware profile"),
        (r"arc_tops\s*=\s*40\.0", "Arc TOPS fallback: 40.0"),
        (r"npu_tops\s*=\s*26\.4", "NPU TOPS fallback: 26.4"),
        (r"ncs2_inference_mb\s*=\s*512\.0", "NCS2 inference fallback: 512 MB"),
        (r"ncs2_count\s*=\s*1", "NCS2 count fallback: 1"),
    ]

    incorrect_patterns = [
        (r"13GB", "Old incorrect RAM: 13 GB"),
        (r"100\.0.*TOPS.*Arc|Arc.*100\.0.*TOPS", "Old incorrect Arc: 100 TOPS"),
        (r"30\.0.*TOPS.*NPU|NPU.*30\.0.*TOPS", "Old incorrect NPU: 30 TOPS"),
        (r"48GB.*NCS2|NCS2.*48GB", "Old incorrect NCS2: 48 GB"),
        (r"1\.5.*GB.*NCS2|NCS2.*1\.5.*GB", "Old incorrect NCS2: 1.5 GB"),
    ]

    success, errors = check_file_for_patterns(filepath, correct_patterns, incorrect_patterns)

    if success:
        print("✅ All corrections verified!")
    else:
        print("❌ Errors found:")
        for error in errors:
            print(error)

    return success


def test_whiterabbitneo_guide():
    """Test WHITERABBITNEO_GUIDE.md corrections."""
    print("\n" + "=" * 70)
    print("TEST 3: WHITERABBITNEO_GUIDE.md")
    print("=" * 70)

    filepath = "/home/user/LAT5150DRVMIL/04-hardware/WHITERABBITNEO_GUIDE.md"

    correct_patterns = [
        (r"62GB system RAM", "62GB RAM mentioned"),
        (r"56GB usable", "56GB usable mentioned"),
        (r"76\.4 TOPS", "76.4 TOPS total"),
        (r"Arc:40", "Arc GPU: 40 TOPS"),
        (r"NPU:26\.4", "NPU: 26.4 TOPS"),
        (r"NCS2:10", "NCS2: 10 TOPS"),
        (r"40 TOPS INT8", "Arc spec: 40 TOPS INT8"),
        (r"26\.4 TOPS.*military mode", "NPU military mode: 26.4 TOPS"),
        (r"10 TOPS.*1 device", "NCS2: 10 TOPS, 1 device"),
        (r"512MB inference memory", "NCS2 inference memory"),
        (r"16GB.*storage.*model caching|16GB on-stick storage", "NCS2 storage clarification"),
    ]

    incorrect_patterns = [
        (r"64GB.*RAM(?!.*62GB)", "Old incorrect RAM: 64 GB (without 62GB correction)"),
        (r"106GB", "Old incorrect total: 106 GB"),
        (r"260 TOPS", "Old incorrect total: 260 TOPS"),
        (r"Arc:100|100 TOPS.*Arc", "Old incorrect Arc: 100 TOPS"),
        (r"NPU:30|30 TOPS.*NPU", "Old incorrect NPU: 30 TOPS"),
        (r"NCS2.*3x|3x.*NCS2", "Old incorrect NCS2: 3 devices"),
        (r"48GB.*NCS2|NCS2.*48GB", "Old incorrect NCS2: 48 GB"),
    ]

    success, errors = check_file_for_patterns(filepath, correct_patterns, incorrect_patterns)

    if success:
        print("✅ All corrections verified!")
    else:
        print("❌ Errors found:")
        for error in errors:
            print(error)

    return success


def test_whiterabbit_demo():
    """Test whiterabbit_demo.py corrections."""
    print("\n" + "=" * 70)
    print("TEST 4: whiterabbit_demo.py")
    print("=" * 70)

    filepath = "/home/user/LAT5150DRVMIL/02-ai-engine/whiterabbit_demo.py"

    correct_patterns = [
        (r"62GB System RAM", "62GB RAM in banner"),
        (r"56GB usable", "56GB usable mentioned"),
        (r"76\.4 TOPS", "76.4 TOPS total"),
        (r"Arc:40", "Arc: 40 TOPS"),
        (r"NPU:26\.4", "NPU: 26.4 TOPS"),
        (r"NCS2:10", "NCS2: 10 TOPS"),
        (r"40 TOPS INT8", "Arc spec: 40 TOPS INT8"),
        (r"26\.4 TOPS.*military", "NPU military mode: 26.4 TOPS"),
        (r"1 device with 512MB inference", "NCS2: 1 device, 512MB"),
        (r"16GB on-stick storage", "NCS2 storage clarification"),
    ]

    incorrect_patterns = [
        (r"64GB RAM(?!.*62GB)", "Old incorrect RAM: 64 GB"),
        (r"106GB", "Old incorrect total: 106 GB"),
        (r"260 TOPS", "Old incorrect total: 260 TOPS"),
        (r"100 TOPS.*compute(?!.*40)", "Old incorrect Arc: 100 TOPS"),
        (r"30 TOPS.*NPU|NPU.*30 TOPS(?!.*26\.4)", "Old incorrect NPU: 30 TOPS"),
        (r"3 devices.*NCS2|NCS2.*3 devices", "Old incorrect NCS2: 3 devices"),
        (r"48GB.*NCS2|NCS2.*48GB", "Old incorrect NCS2: 48 GB"),
    ]

    success, errors = check_file_for_patterns(filepath, correct_patterns, incorrect_patterns)

    if success:
        print("✅ All corrections verified!")
    else:
        print("❌ Errors found:")
        for error in errors:
            print(error)

    return success


def main():
    """Run all verification tests."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║         WhiteRabbitNeo Specification Correction Verification         ║
║                                                                      ║
║  Verifying all files have been corrected with accurate specs        ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

    tests = [
        test_hardware_profile,
        test_dynamic_allocator,
        test_whiterabbitneo_guide,
        test_whiterabbit_demo,
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
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    passed = sum(results)
    total = len(results)

    print(f"\nPassed: {passed}/{total}")

    if all(results):
        print("\n✅ ALL VERIFICATIONS PASSED!")
        print("\n📋 Corrected Specifications Summary:")
        print("  ┌─────────────────────────────────────────────┐")
        print("  │ Component       │ Old Value  │ New Value   │")
        print("  ├─────────────────────────────────────────────┤")
        print("  │ System RAM      │ 64 GB      │ 62 GB       │")
        print("  │ Usable RAM      │ 58-60 GB   │ 56 GB       │")
        print("  │ Arc GPU TOPS    │ 100 TOPS   │ 40 TOPS     │")
        print("  │ NPU TOPS        │ 30 TOPS    │ 26.4 TOPS   │")
        print("  │ NCS2 Devices    │ 3 devices  │ 1 device    │")
        print("  │ NCS2 Inference  │ 48 GB      │ 512 MB      │")
        print("  │ NCS2 Storage    │ (confused) │ 16 GB       │")
        print("  │ NCS2 TOPS       │ 30 TOPS    │ 10 TOPS     │")
        print("  │ Total Memory    │ 106 GB     │ 62 GB       │")
        print("  │ Total TOPS      │ 260 TOPS   │ 76.4 TOPS   │")
        print("  └─────────────────────────────────────────────┘")
        print("\n✨ All files corrected successfully!")
        return 0
    else:
        print("\n❌ SOME VERIFICATIONS FAILED")
        failed_tests = [i for i, r in enumerate(results) if not r]
        print(f"Failed tests: {failed_tests}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
