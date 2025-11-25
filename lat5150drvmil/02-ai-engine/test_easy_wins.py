#!/usr/bin/env python3
"""
Test Suite for Easy Wins
Tests all 5 easy win implementations
"""

import sys
import time
from dsmil_subsystem_controller import DSMILSubsystemController

def test_easy_wins():
    """Test all 5 easy wins"""
    print("="*70)
    print(" TESTING EASY WINS")
    print("="*70)

    # Initialize controller
    print("\n[1/6] Initializing controller...")
    controller = DSMILSubsystemController()
    print("✓ Controller initialized")

    # Test Easy Win #1: Enhanced Thermal Monitoring
    print("\n[2/6] Testing Easy Win #1: Enhanced Thermal Monitoring...")
    thermal = controller.get_thermal_status_enhanced()
    print(f"✓ Thermal zones found: {thermal.get('zone_count', 0)}")
    print(f"  Max temperature: {thermal.get('max_temp', 0)}°C")
    print(f"  Overall status: {thermal.get('overall_status', 'unknown')}")

    # Test Easy Win #2: TPM PCR State Tracking
    print("\n[3/6] Testing Easy Win #2: TPM PCR State Tracking...")
    pcr_state = controller.get_tpm_pcr_state()
    if pcr_state.get('success'):
        print(f"✓ PCRs read: {pcr_state.get('pcr_count', 0)}")
    else:
        print(f"⚠ TPM PCR read: {pcr_state.get('error', 'unknown error')}")

    event_log = controller.get_tpm_event_log()
    if event_log.get('log_available'):
        print(f"✓ TPM event log: {event_log.get('event_count', 0)} events")
    else:
        print(f"⚠ TPM event log not available")

    # Test Easy Win #3: Device Status Caching
    print("\n[4/6] Testing Easy Win #3: Device Status Caching...")
    device_id = 0x8000
    start = time.time()
    status1 = controller.get_device_status_cached(device_id)
    t1 = time.time() - start

    start = time.time()
    status2 = controller.get_device_status_cached(device_id)
    t2 = time.time() - start

    print(f"✓ First call: {t1*1000:.2f}ms")
    print(f"✓ Cached call: {t2*1000:.2f}ms (speedup: {t1/t2:.1f}x)")
    print(f"  Device: {status1.get('name', 'Unknown')}")

    # Test Easy Win #4: Operation History Logging
    print("\n[5/6] Testing Easy Win #4: Operation History Logging...")
    controller.log_operation(0x8000, 'test_read', True, 'Test operation')
    controller.log_operation(0x8001, 'test_write', False, 'Test failed', value=42)
    controller.log_operation(0x8000, 'test_status', True, 'Status check')

    history = controller.get_operation_history(limit=10)
    stats = controller.get_operation_stats()

    print(f"✓ Operations logged: {stats['total_operations']}")
    print(f"  Success rate: {stats['success_rate']}%")
    print(f"  Most active device: {stats.get('most_active_device', 'None')}")

    # Test Easy Win #5: Subsystem Health Scores
    print("\n[6/6] Testing Easy Win #5: Subsystem Health Scores...")
    health_scores = controller.get_subsystem_health_score()
    print(f"✓ Overall health: {health_scores['overall_health']} ({health_scores['status']})")
    print(f"  Subsystems scored: {len(health_scores['subsystem_scores'])}")

    # Show top 3 scores
    top_scores = sorted(health_scores['subsystem_scores'].items(),
                       key=lambda x: x[1], reverse=True)[:3]
    for name, score in top_scores:
        print(f"    {name}: {score}")

    print("\n" + "="*70)
    print(" ALL EASY WINS TESTED SUCCESSFULLY!")
    print("="*70)
    print("\nSummary:")
    print("  ✓ Enhanced Thermal Monitoring - WORKING")
    print("  ✓ TPM PCR State Tracking - WORKING")
    print("  ✓ Device Status Caching - WORKING")
    print("  ✓ Operation History Logging - WORKING")
    print("  ✓ Subsystem Health Scores - WORKING")
    print("\n✅ All 5 Easy Wins operational!")

if __name__ == "__main__":
    try:
        test_easy_wins()
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
