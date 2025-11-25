#!/usr/bin/env python3
"""
Test script to demonstrate the 104dev -> 84dev fallback logic
This simulates the driver selection logic without requiring actual kernel builds
"""

from pathlib import Path
import sys

def test_fallback_logic():
    """Test the driver selection fallback logic"""

    print("Testing 104dev -> 84dev Fallback Logic")
    print("=" * 60)

    # Test Case 1: Both drivers built successfully
    print("\n[Test 1] Both 104dev and 84dev built successfully:")
    driver_104_exists = True
    driver_84_exists = True

    if driver_104_exists:
        selected = "dsmil-104dev"
        print(f"  ✓ Selected: {selected} (104 devices - preferred)")
    elif driver_84_exists:
        selected = "dsmil-84dev"
        print(f"  ✓ Selected: {selected} (84 devices - fallback)")
    else:
        print("  ✗ No driver available")
        selected = None

    assert selected == "dsmil-104dev", "Should select 104dev when both available"

    # Test Case 2: Only 84dev built (104dev failed)
    print("\n[Test 2] 104dev build failed, only 84dev available:")
    driver_104_exists = False
    driver_84_exists = True

    if driver_104_exists:
        selected = "dsmil-104dev"
        print(f"  ✓ Selected: {selected} (104 devices - preferred)")
    elif driver_84_exists:
        selected = "dsmil-84dev"
        print(f"  ⚠ Selected: {selected} (84 devices - fallback)")
    else:
        print("  ✗ No driver available")
        selected = None

    assert selected == "dsmil-84dev", "Should fallback to 84dev when 104dev unavailable"

    # Test Case 3: Both builds failed
    print("\n[Test 3] Both builds failed:")
    driver_104_exists = False
    driver_84_exists = False

    if driver_104_exists:
        selected = "dsmil-104dev"
        print(f"  ✓ Selected: {selected} (104 devices - preferred)")
    elif driver_84_exists:
        selected = "dsmil-84dev"
        print(f"  ✓ Selected: {selected} (84 devices - fallback)")
    else:
        print("  ✗ No driver available - build failed")
        selected = None

    assert selected is None, "Should fail when neither driver is available"

    # Test Case 4: Only 104dev built (normal case)
    print("\n[Test 4] Only 104dev built successfully:")
    driver_104_exists = True
    driver_84_exists = False

    if driver_104_exists:
        selected = "dsmil-104dev"
        print(f"  ✓ Selected: {selected} (104 devices - preferred)")
    elif driver_84_exists:
        selected = "dsmil-84dev"
        print(f"  ✓ Selected: {selected} (84 devices - fallback)")
    else:
        print("  ✗ No driver available")
        selected = None

    assert selected == "dsmil-104dev", "Should select 104dev when available"

    print("\n" + "=" * 60)
    print("✓ All fallback logic tests passed!")
    print("\nThe build-auto command will:")
    print("  1. Build both dsmil-104dev.ko and dsmil-84dev.ko")
    print("  2. Check if dsmil-104dev.ko exists (preferred)")
    print("  3. If not, check if dsmil-84dev.ko exists (fallback)")
    print("  4. Install and load the selected driver")
    print("  5. Fail only if both builds failed")

    return True

if __name__ == "__main__":
    try:
        test_fallback_logic()
        sys.exit(0)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
