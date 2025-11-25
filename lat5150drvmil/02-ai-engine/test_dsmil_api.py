#!/usr/bin/env python3
"""
Test DSMIL API Endpoints

Tests all DSMIL subsystem control endpoints to verify functionality.
"""

import json
import sys
import requests
import time
from datetime import datetime

# API base URL
BASE_URL = "http://localhost:5050"

def test_endpoint(name, method, endpoint, data=None, expected_status=200):
    """Test a single API endpoint"""
    url = f"{BASE_URL}{endpoint}"

    print(f"\n{'='*70}")
    print(f"TEST: {name}")
    print(f"{'='*70}")
    print(f"Method: {method}")
    print(f"Endpoint: {endpoint}")

    if data:
        print(f"Data: {json.dumps(data, indent=2)}")

    try:
        if method == "GET":
            response = requests.get(url, timeout=5)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=5)
        else:
            print(f"❌ Unsupported method: {method}")
            return False

        print(f"\nStatus Code: {response.status_code}")

        # Parse response
        try:
            response_data = response.json()
            print(f"Response:\n{json.dumps(response_data, indent=2)}")
        except:
            print(f"Response: {response.text}")

        # Check status code
        if response.status_code == expected_status:
            print(f"\n✅ PASS: Status code matches expected {expected_status}")
            return True
        else:
            print(f"\n❌ FAIL: Expected {expected_status}, got {response.status_code}")
            return False

    except requests.exceptions.ConnectionError:
        print(f"❌ FAIL: Cannot connect to {BASE_URL}")
        print("   Make sure the GUI dashboard is running!")
        return False
    except requests.exceptions.Timeout:
        print(f"❌ FAIL: Request timed out")
        return False
    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False


def main():
    """Run all DSMIL API tests"""

    print("\n" + "="*70)
    print(" DSMIL API ENDPOINT TESTING")
    print("="*70)
    print(f"Target: {BASE_URL}")
    print(f"Time: {datetime.now().isoformat()}")
    print("="*70)

    results = []

    # Test 1: System Health
    results.append(test_endpoint(
        "DSMIL System Health",
        "GET",
        "/api/dsmil/health"
    ))

    time.sleep(0.5)

    # Test 2: All Subsystems Status
    results.append(test_endpoint(
        "All Subsystems Status",
        "GET",
        "/api/dsmil/subsystems"
    ))

    time.sleep(0.5)

    # Test 3: List Safe Devices
    results.append(test_endpoint(
        "List Safe Devices",
        "GET",
        "/api/dsmil/devices/safe"
    ))

    time.sleep(0.5)

    # Test 4: List Quarantined Devices
    results.append(test_endpoint(
        "List Quarantined Devices",
        "GET",
        "/api/dsmil/devices/quarantined"
    ))

    time.sleep(0.5)

    # Test 5: Activate Safe Device (0x8003 - Display)
    results.append(test_endpoint(
        "Activate Safe Device (0x8003)",
        "POST",
        "/api/dsmil/device/activate",
        data={"device_id": "0x8003", "value": 1}
    ))

    time.sleep(0.5)

    # Test 6: Attempt to Activate Quarantined Device (should fail)
    results.append(test_endpoint(
        "Activate Quarantined Device (0x8009) - SHOULD FAIL",
        "POST",
        "/api/dsmil/device/activate",
        data={"device_id": "0x8009", "value": 1},
        expected_status=403  # Expect forbidden
    ))

    time.sleep(0.5)

    # Test 7: TPM Quote
    results.append(test_endpoint(
        "Get TPM Quote",
        "GET",
        "/api/dsmil/tpm/quote"
    ))

    time.sleep(0.5)

    # Test 8: Comprehensive Metrics
    results.append(test_endpoint(
        "Get Comprehensive Metrics",
        "GET",
        "/api/dsmil/metrics"
    ))

    # Summary
    print("\n" + "="*70)
    print(" TEST SUMMARY")
    print("="*70)

    passed = sum(results)
    total = len(results)

    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")

    if passed == total:
        print("\n✅ ALL TESTS PASSED")
        return 0
    else:
        print(f"\n❌ {total - passed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
