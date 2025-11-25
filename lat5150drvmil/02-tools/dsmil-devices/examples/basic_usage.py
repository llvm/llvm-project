#!/usr/bin/env python3
"""
DSMIL Device Integration - Basic Usage Example

Demonstrates basic operations with integrated DSMIL devices.

Author: DSMIL Integration Framework
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dsmil_integration import (
    get_device, get_all_devices, initialize_all_devices,
    print_integration_summary, print_device_list
)


def example_tpm_control():
    """Example: TPM Control (0x8000)"""
    print("\n" + "=" * 80)
    print("Example: TPM Control (0x8000)")
    print("=" * 80)

    tpm = get_device(0x8000)
    if not tpm:
        print("TPM Control device not available")
        return

    # Initialize
    result = tpm.initialize()
    print(f"\n1. Initialize TPM: {result.success}")
    if result.success:
        print(f"   Version: {result.data['version']}")
        print(f"   Algorithms: {result.data['algorithms']}")

    # Get status
    status = tpm.get_status()
    print(f"\n2. TPM Status:")
    print(f"   Ready: {status['ready']}")
    print(f"   Version: {status['version']}")
    print(f"   Active keys: {status['active_keys']}")

    # Read capabilities
    result = tpm.read_register("CAPABILITIES")
    print(f"\n3. Read CAPABILITIES register: {result.success}")
    if result.success:
        print(f"   Value: {result.data['hex']}")

    # Generate random bytes
    result = tpm.get_random(32)
    print(f"\n4. Generate 32 random bytes: {result.success}")
    if result.success:
        print(f"   Data: {result.data['data'][:32]}...")

    # Read PCR
    result = tpm.read_pcr(0)
    print(f"\n5. Read PCR[0]: {result.success}")
    if result.success:
        print(f"   Value: {result.data['hex'][:32]}...")


def example_boot_security():
    """Example: Boot Security (0x8001)"""
    print("\n" + "=" * 80)
    print("Example: Boot Security (0x8001)")
    print("=" * 80)

    boot_sec = get_device(0x8001)
    if not boot_sec:
        print("Boot Security device not available")
        return

    # Initialize
    result = boot_sec.initialize()
    print(f"\n1. Initialize Boot Security: {result.success}")

    # Get boot policy
    result = boot_sec.get_boot_policy()
    print(f"\n2. Boot Policy: {result.success}")
    if result.success:
        print(f"   Flags: {', '.join(result.data['flags'])}")

    # Get boot measurements
    result = boot_sec.get_boot_measurements()
    print(f"\n3. Boot Measurements: {result.success}")
    if result.success:
        print(f"   Total: {result.data['total']}")
        print(f"   All valid: {result.data['all_valid']}")

    # Get boot chain summary
    result = boot_sec.get_boot_chain_summary()
    print(f"\n4. Boot Chain Summary:")
    if result.success:
        for key, value in result.data.items():
            print(f"   {key}: {value}")


def example_credential_vault():
    """Example: Credential Vault (0x8002)"""
    print("\n" + "=" * 80)
    print("Example: Credential Vault (0x8002)")
    print("=" * 80)

    vault = get_device(0x8002)
    if not vault:
        print("Credential Vault device not available")
        return

    # Initialize
    result = vault.initialize()
    print(f"\n1. Initialize Vault: {result.success}")
    if result.success:
        print(f"   Locked: {result.data['locked']}")
        print(f"   Credentials: {result.data['credentials']}")

    # Try to list credentials while locked
    result = vault.list_credentials()
    print(f"\n2. List credentials (locked): {result.success}")
    if not result.success:
        print(f"   Error: {result.error}")

    # Unlock vault
    result = vault.unlock_vault(auth_token="simulated_token")
    print(f"\n3. Unlock vault: {result.success}")

    # List credentials after unlock
    if result.success:
        result = vault.list_credentials()
        print(f"\n4. List credentials (unlocked): {result.success}")
        if result.success:
            for cred in result.data['credentials']:
                print(f"   Slot {cred['slot']}: {cred['name']} ({cred['type']})")

    # Get capacity info
    result = vault.get_capacity_info()
    print(f"\n5. Capacity info: {result.success}")
    if result.success:
        print(f"   Used: {result.data['used_slots']}/{result.data['total_slots']}")
        print(f"   Usage: {result.data['usage_percent']}%")

    # Lock vault
    vault.lock_vault()
    print(f"\n6. Lock vault: done")


def example_intrusion_detection():
    """Example: Intrusion Detection (0x8010)"""
    print("\n" + "=" * 80)
    print("Example: Intrusion Detection (0x8010)")
    print("=" * 80)

    ids = get_device(0x8010)
    if not ids:
        print("Intrusion Detection device not available")
        return

    # Initialize
    result = ids.initialize()
    print(f"\n1. Initialize IDS: {result.success}")
    if result.success:
        print(f"   Armed: {result.data['armed']}")
        print(f"   All secure: {result.data['all_secure']}")

    # Get sensor states
    result = ids.get_sensor_states()
    print(f"\n2. Sensor states: {result.success}")
    if result.success:
        for sensor in result.data['sensors']:
            print(f"   {sensor['sensor']}: {sensor['status']}")

    # Get status
    status = ids.get_status()
    print(f"\n3. IDS Status:")
    for key, value in status.items():
        if key != 'state':
            print(f"   {key}: {value}")


def main():
    """Main example runner"""
    print("=" * 80)
    print("DSMIL Device Integration - Basic Usage Examples")
    print("=" * 80)

    # Show summary
    print_integration_summary()

    # Initialize all devices
    print("\n" + "=" * 80)
    print("Initializing all devices...")
    print("=" * 80)

    results = initialize_all_devices()
    for device_id, result in results.items():
        status = "✓" if result.success else "✗"
        print(f"{status} Device 0x{device_id:04X}: {'SUCCESS' if result.success else result.error}")

    # Run examples
    example_tpm_control()
    example_boot_security()
    example_credential_vault()
    example_intrusion_detection()

    # Show device list
    print("\n")
    print_device_list()

    print("\n" + "=" * 80)
    print("Examples complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
