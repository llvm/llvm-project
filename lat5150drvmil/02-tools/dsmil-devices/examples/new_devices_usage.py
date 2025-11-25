#!/usr/bin/env python3
"""
DSMIL Device Integration - New Devices Usage Examples

Demonstrates usage of the 5 newly integrated devices:
- 0x8005: Performance Monitor / TPM-HSM Interface
- 0x8016: VPN Controller
- 0x801E: Tactical Display Control
- 0x8050: Storage Encryption Controller
- 0x805A: Sensor Array Controller

Author: DSMIL Integration Framework
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dsmil_integration import (
    get_device, initialize_all_devices, print_integration_summary
)


def example_performance_monitor():
    """Example: Performance Monitor (0x8005)"""
    print("\n" + "=" * 80)
    print("Example: Performance Monitor / TPM-HSM Interface (0x8005)")
    print("=" * 80)

    perf_mon = get_device(0x8005)
    if not perf_mon:
        print("Performance Monitor device not available")
        return

    # Initialize
    result = perf_mon.initialize()
    print(f"\n1. Initialize Performance Monitor: {result.success}")
    if result.success:
        print(f"   Active: {result.data['monitoring_active']}")
        print(f"   Sample interval: {result.data['sample_interval']}s")

    # Get current metrics
    result = perf_mon.get_current_metrics()
    print(f"\n2. Current Metrics: {result.success}")
    if result.success:
        metrics = result.data['metrics']
        print(f"   CPU Usage: {metrics['cpu_usage']:.1f}%")
        print(f"   Memory Usage: {metrics['memory_usage']:.1f}%")
        print(f"   Crypto Ops/s: {metrics['crypto_ops']}")
        print(f"   TPM Activity: {metrics['tpm_activity']} ops/s")

    # Get thermal status
    result = perf_mon.get_thermal_status()
    print(f"\n3. Thermal Status: {result.success}")
    if result.success:
        print(f"   Temperature: {result.data['temperature_celsius']:.1f}°C")
        print(f"   Status: {result.data['status']}")

    # Get HSM status
    result = perf_mon.get_hsm_status()
    print(f"\n4. HSM Status: {result.success}")
    if result.success:
        print(f"   Connected: {result.data['connected']}")
        print(f"   Status: {result.data['status']}")


def example_vpn_controller():
    """Example: VPN Controller (0x8016)"""
    print("\n" + "=" * 80)
    print("Example: VPN Controller (0x8016)")
    print("=" * 80)

    vpn = get_device(0x8016)
    if not vpn:
        print("VPN Controller device not available")
        return

    # Initialize
    result = vpn.initialize()
    print(f"\n1. Initialize VPN Controller: {result.success}")
    if result.success:
        print(f"   VPN enabled: {result.data['vpn_enabled']}")
        print(f"   FIPS mode: {result.data['fips_mode']}")

    # List tunnels
    result = vpn.list_tunnels()
    print(f"\n2. List VPN Tunnels: {result.success}")
    if result.success:
        print(f"   Total tunnels: {result.data['total']}")
        for tunnel in result.data.get('tunnels', []):
            print(f"   - {tunnel['name']}: {tunnel['status']} ({tunnel['protocol']})")

    # Get encryption info
    result = vpn.get_encryption_info()
    print(f"\n3. Encryption Configuration: {result.success}")
    if result.success:
        print(f"   FIPS mode: {result.data['fips_mode']}")
        print(f"   Key exchange: {result.data['key_exchange']}")
        print(f"   Allowed ciphers: {', '.join(result.data['allowed_ciphers'][:2])}")

    # Get status
    status = vpn.get_status()
    print(f"\n4. VPN Status:")
    print(f"   VPN enabled: {status['vpn_enabled']}")
    print(f"   Encryption OK: {status['encryption_ok']}")
    print(f"   FIPS mode: {status['fips_mode']}")


def example_tactical_display():
    """Example: Tactical Display Control (0x801E)"""
    print("\n" + "=" * 80)
    print("Example: Tactical Display Control (0x801E)")
    print("=" * 80)

    display = get_device(0x801E)
    if not display:
        print("Tactical Display device not available")
        return

    # Initialize
    result = display.initialize()
    print(f"\n1. Initialize Tactical Display: {result.success}")
    if result.success:
        print(f"   Displays: {result.data['displays']}")
        print(f"   Security zone: {result.data['security_zone']}")
        print(f"   Display mode: {result.data['display_mode']}")
        print(f"   TEMPEST compliant: {result.data['tempest_compliant']}")

    # List displays
    result = display.list_displays()
    print(f"\n2. Display List: {result.success}")
    if result.success:
        print(f"   Total displays: {result.data['total']}")
        for disp in result.data.get('displays', []):
            print(f"   - {disp['name']}: {disp['resolution']} ({'active' if disp['active'] else 'inactive'})")

    # Get security config
    result = display.get_security_config()
    print(f"\n3. Security Configuration: {result.success}")
    if result.success:
        print(f"   Security zone: {result.data['security_zone']}")
        print(f"   Protection level: {result.data['protection_level']}")
        print(f"   Capture blocking: {result.data['capture_blocking']}")
        print(f"   Watermarking: {result.data['watermarking']}")

    # Get TEMPEST status
    result = display.get_tempest_status()
    print(f"\n4. TEMPEST Status: {result.success}")
    if result.success:
        print(f"   Compliant: {result.data['compliant']}")
        print(f"   Emission level: {result.data['emission_level']}")
        print(f"   Certification: {result.data.get('certification', 'None')}")


def example_storage_encryption():
    """Example: Storage Encryption Controller (0x8050)"""
    print("\n" + "=" * 80)
    print("Example: Storage Encryption Controller (0x8050)")
    print("=" * 80)

    storage = get_device(0x8050)
    if not storage:
        print("Storage Encryption device not available")
        return

    # Initialize
    result = storage.initialize()
    print(f"\n1. Initialize Storage Encryption: {result.success}")
    if result.success:
        print(f"   FDE enabled: {result.data['fde_enabled']}")
        print(f"   FIPS mode: {result.data['fips_mode']}")
        print(f"   Algorithm: {result.data['algorithm']}")

    # List volumes
    result = storage.list_volumes()
    print(f"\n2. Volume List: {result.success}")
    if result.success:
        print(f"   Total volumes: {result.data['total']}")
        print(f"   Encrypted: {result.data['encrypted']}")
        for volume in result.data.get('volumes', []):
            print(f"   - {volume['name']}: {volume['size_gb']}GB - {volume['status']}")

    # Get encryption config
    result = storage.get_encryption_config()
    print(f"\n3. Encryption Configuration: {result.success}")
    if result.success:
        print(f"   FDE enabled: {result.data['fde_enabled']}")
        print(f"   Default algorithm: {result.data['default_algorithm']}")
        print(f"   Key strength: {result.data['key_strength']} bits")
        print(f"   Hardware crypto: {result.data['hardware_crypto']}")

    # List SED drives
    result = storage.list_sed_drives()
    print(f"\n4. Self-Encrypting Drives: {result.success}")
    if result.success:
        print(f"   Total SEDs: {result.data['total']}")
        for drive in result.data.get('drives', []):
            print(f"   - {drive['model']}: {drive['capacity_gb']}GB - {drive['status']}")

    # Get OPAL support
    result = storage.get_opal_support()
    print(f"\n5. OPAL Support: {result.success}")
    if result.success:
        print(f"   Supported: {result.data['supported']}")
        if result.data['supported']:
            print(f"   Version: {result.data['version']}")
            print(f"   Features: {len(result.data['features'])} available")


def example_sensor_array():
    """Example: Sensor Array Controller (0x805A)"""
    print("\n" + "=" * 80)
    print("Example: Sensor Array Controller (0x805A)")
    print("=" * 80)

    sensors = get_device(0x805A)
    if not sensors:
        print("Sensor Array device not available")
        return

    # Initialize
    result = sensors.initialize()
    print(f"\n1. Initialize Sensor Array: {result.success}")
    if result.success:
        print(f"   Total sensors: {result.data['total_sensors']}")
        print(f"   Active sensors: {result.data['active_sensors']}")
        print(f"   Fusion enabled: {result.data['fusion_enabled']}")

    # List sensors
    result = sensors.list_sensors()
    print(f"\n2. Sensor List: {result.success}")
    if result.success:
        print(f"   Total: {result.data['total']}, Online: {result.data['online']}")
        for sensor in result.data.get('sensors', [])[:5]:  # Show first 5
            print(f"   - {sensor['name']}: {sensor['value']} {sensor['unit']} ({sensor['status']})")

    # Get environmental summary
    result = sensors.get_environmental_summary()
    print(f"\n3. Environmental Summary: {result.success}")
    if result.success:
        temp = result.data['temperature']
        humid = result.data['humidity']
        print(f"   Temperature: {temp['celsius']:.1f}°C ({temp['status']})")
        print(f"   Humidity: {humid['percent']:.1f}% ({humid['status']})")
        print(f"   Light: {result.data['light']['lux']:.0f} lux ({result.data['light']['condition']})")

    # Get security summary
    result = sensors.get_security_summary()
    print(f"\n4. Security Summary: {result.success}")
    if result.success:
        print(f"   Overall status: {result.data['overall_status']}")
        print(f"   Motion detected: {result.data['motion_detected']}")
        print(f"   Tamper detected: {result.data['tamper_detected']}")
        print(f"   Vibration detected: {result.data['vibration_detected']}")

    # Get radiation status
    result = sensors.get_radiation_status()
    print(f"\n5. Radiation Status: {result.success}")
    if result.success:
        print(f"   Level: {result.data['level_usv_per_hour']:.2f} μSv/h")
        print(f"   Status: {result.data['status']}")

    # Get sensor fusion data
    result = sensors.get_fusion_data()
    print(f"\n6. Sensor Fusion: {result.success}")
    if result.success:
        sa = result.data['situational_awareness']
        print(f"   Environment stable: {sa['environment_stable']}")
        print(f"   Security status: {sa['security_status']}")
        print(f"   Threat level: {sa['threat_level']}")
        print(f"   Confidence: {sa['confidence']:.0%}")


def main():
    """Main example runner"""
    print("=" * 80)
    print("DSMIL New Devices - Usage Examples")
    print("=" * 80)

    # Show integration summary
    print_integration_summary()

    # Initialize all devices
    print("\n" + "=" * 80)
    print("Initializing all devices...")
    print("=" * 80)

    results = initialize_all_devices()

    # Count new devices
    new_device_ids = [0x8005, 0x8016, 0x801E, 0x8050, 0x805A]
    for device_id in new_device_ids:
        if device_id in results:
            result = results[device_id]
            status = "✓" if result.success else "✗"
            print(f"{status} Device 0x{device_id:04X}: {'SUCCESS' if result.success else result.error}")

    # Run examples for new devices
    example_performance_monitor()
    example_vpn_controller()
    example_tactical_display()
    example_storage_encryption()
    example_sensor_array()

    print("\n" + "=" * 80)
    print("All examples complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
