#!/usr/bin/env python3
"""
DSMIL Device Integration Module

Unified interface for all integrated DSMIL devices. Provides centralized
device registration, initialization, and access control.

Author: DSMIL Integration Framework
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
"""

import sys
import os

# Add lib directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'devices'))

from lib.device_registry import (
    DSMILDeviceRegistry, get_registry, DeviceGroup, DeviceRiskLevel
)

# Import all device implementations
from device_0x8000_tpm_control import TPMControlDevice
from device_0x8001_boot_security import BootSecurityDevice
from device_0x8002_credential_vault import CredentialVaultDevice
from device_0x8003_audit_log import AuditLogDevice
from device_0x8004_event_logger import EventLoggerDevice
from device_0x8005_performance_monitor import PerformanceMonitorDevice
from device_0x8006_thermal_sensor import ThermalSensorDevice
from device_0x8007_power_state import PowerStateDevice
from device_0x8008_emergency_response import EmergencyResponseDevice
from device_0x8010_intrusion_detection import IntrusionDetectionDevice
from device_0x8013_key_management import KeyManagementDevice
from device_0x8014_certificate_store import CertificateStoreDevice
from device_0x8016_vpn_controller import VPNControllerDevice
from device_0x8017_remote_access import RemoteAccessDevice
from device_0x8018_pre_isolation import PreIsolationDevice
from device_0x801A_port_security import PortSecurityDevice
from device_0x801B_wireless_security import WirelessSecurityDevice
from device_0x801E_tactical_display import TacticalDisplayDevice
from device_0x802A_network_monitor import NetworkMonitorDevice
from device_0x802B_packet_filter import PacketFilterDevice
from device_0x8050_storage_encryption import StorageEncryptionDevice
from device_0x805A_sensor_array import SensorArrayDevice


# Version information
VERSION = "1.5.0"
INTEGRATION_NAME = "DSMIL Device Integration"


def register_all_devices():
    """
    Register all integrated devices with the global registry.

    This includes 22 integrated devices across multiple groups:
    - Group 0 (Core Security): 9 devices
    - Group 1 (Extended Security): 8 devices
    - Group 2 (Network/Comms): 3 devices
    - Group 4 (Storage/Management): 1 device
    - Group 5 (Peripheral/Control): 1 device

    Returns:
        DSMILDeviceRegistry: The global device registry
    """
    registry = get_registry()

    # Group 0: Core Security (9 devices)
    registry.register_device(
        device_id=0x8000,
        device_class=TPMControlDevice,
        risk_level=DeviceRiskLevel.MONITORED,
        group=DeviceGroup.GROUP_0_CORE_SECURITY,
        enabled=True
    )

    registry.register_device(
        device_id=0x8001,
        device_class=BootSecurityDevice,
        risk_level=DeviceRiskLevel.MONITORED,
        group=DeviceGroup.GROUP_0_CORE_SECURITY,
        enabled=True
    )

    registry.register_device(
        device_id=0x8002,
        device_class=CredentialVaultDevice,
        risk_level=DeviceRiskLevel.MONITORED,
        group=DeviceGroup.GROUP_0_CORE_SECURITY,
        enabled=True
    )

    registry.register_device(
        device_id=0x8003,
        device_class=AuditLogDevice,
        risk_level=DeviceRiskLevel.SAFE,
        group=DeviceGroup.GROUP_0_CORE_SECURITY,
        enabled=True
    )

    registry.register_device(
        device_id=0x8004,
        device_class=EventLoggerDevice,
        risk_level=DeviceRiskLevel.SAFE,
        group=DeviceGroup.GROUP_0_CORE_SECURITY,
        enabled=True
    )

    registry.register_device(
        device_id=0x8005,
        device_class=PerformanceMonitorDevice,
        risk_level=DeviceRiskLevel.SAFE,
        group=DeviceGroup.GROUP_0_CORE_SECURITY,
        enabled=True
    )

    registry.register_device(
        device_id=0x8006,
        device_class=ThermalSensorDevice,
        risk_level=DeviceRiskLevel.SAFE,
        group=DeviceGroup.GROUP_0_CORE_SECURITY,
        enabled=True
    )

    registry.register_device(
        device_id=0x8007,
        device_class=PowerStateDevice,
        risk_level=DeviceRiskLevel.MONITORED,
        group=DeviceGroup.GROUP_0_CORE_SECURITY,
        enabled=True
    )

    registry.register_device(
        device_id=0x8008,
        device_class=EmergencyResponseDevice,
        risk_level=DeviceRiskLevel.MONITORED,
        group=DeviceGroup.GROUP_0_CORE_SECURITY,
        enabled=True
    )

    # Group 1: Extended Security (8 devices)
    registry.register_device(
        device_id=0x8010,
        device_class=IntrusionDetectionDevice,
        risk_level=DeviceRiskLevel.SAFE,
        group=DeviceGroup.GROUP_1_EXTENDED_SECURITY,
        enabled=True
    )

    registry.register_device(
        device_id=0x8013,
        device_class=KeyManagementDevice,
        risk_level=DeviceRiskLevel.MONITORED,
        group=DeviceGroup.GROUP_1_EXTENDED_SECURITY,
        enabled=True
    )

    registry.register_device(
        device_id=0x8014,
        device_class=CertificateStoreDevice,
        risk_level=DeviceRiskLevel.SAFE,
        group=DeviceGroup.GROUP_1_EXTENDED_SECURITY,
        enabled=True
    )

    registry.register_device(
        device_id=0x8016,
        device_class=VPNControllerDevice,
        risk_level=DeviceRiskLevel.MONITORED,
        group=DeviceGroup.GROUP_1_EXTENDED_SECURITY,
        enabled=True
    )

    registry.register_device(
        device_id=0x8017,
        device_class=RemoteAccessDevice,
        risk_level=DeviceRiskLevel.MONITORED,
        group=DeviceGroup.GROUP_1_EXTENDED_SECURITY,
        enabled=True
    )

    registry.register_device(
        device_id=0x8018,
        device_class=PreIsolationDevice,
        risk_level=DeviceRiskLevel.MONITORED,
        group=DeviceGroup.GROUP_1_EXTENDED_SECURITY,
        enabled=True
    )

    registry.register_device(
        device_id=0x801A,
        device_class=PortSecurityDevice,
        risk_level=DeviceRiskLevel.MONITORED,
        group=DeviceGroup.GROUP_1_EXTENDED_SECURITY,
        enabled=True
    )

    registry.register_device(
        device_id=0x801B,
        device_class=WirelessSecurityDevice,
        risk_level=DeviceRiskLevel.MONITORED,
        group=DeviceGroup.GROUP_1_EXTENDED_SECURITY,
        enabled=True
    )

    # Group 2: Network/Communications (3 devices)
    registry.register_device(
        device_id=0x801E,
        device_class=TacticalDisplayDevice,
        risk_level=DeviceRiskLevel.MONITORED,
        group=DeviceGroup.GROUP_2_NETWORK_COMMS,
        enabled=True
    )

    registry.register_device(
        device_id=0x802A,
        device_class=NetworkMonitorDevice,
        risk_level=DeviceRiskLevel.SAFE,
        group=DeviceGroup.GROUP_2_NETWORK_COMMS,
        enabled=True
    )

    registry.register_device(
        device_id=0x802B,
        device_class=PacketFilterDevice,
        risk_level=DeviceRiskLevel.MONITORED,
        group=DeviceGroup.GROUP_2_NETWORK_COMMS,
        enabled=True
    )

    # Group 4: Storage/Management (1 device)
    registry.register_device(
        device_id=0x8050,
        device_class=StorageEncryptionDevice,
        risk_level=DeviceRiskLevel.MONITORED,
        group=DeviceGroup.GROUP_4_STORAGE_MANAGEMENT,
        enabled=True
    )

    # Group 5: Peripheral/Control (1 device)
    registry.register_device(
        device_id=0x805A,
        device_class=SensorArrayDevice,
        risk_level=DeviceRiskLevel.SAFE,
        group=DeviceGroup.GROUP_5_PERIPHERAL_CONTROL,
        enabled=True
    )

    return registry


def initialize_all_devices():
    """
    Initialize all registered devices.

    Returns:
        dict: Dictionary mapping device IDs to initialization results
    """
    registry = get_registry()
    return registry.initialize_all()


def get_device(device_id: int):
    """
    Get a device instance by ID.

    Args:
        device_id: Device ID (0x8000-0x806B)

    Returns:
        DSMILDeviceBase instance or None
    """
    registry = get_registry()
    return registry.get_device(device_id)


def get_all_devices():
    """
    Get all registered devices.

    Returns:
        dict: Dictionary mapping device IDs to device instances
    """
    registry = get_registry()
    return registry.get_all_devices()


def get_devices_by_group(group: DeviceGroup):
    """
    Get all devices in a specific group.

    Args:
        group: DeviceGroup enum value

    Returns:
        list: List of device instances in the group
    """
    registry = get_registry()
    return registry.get_devices_by_group(group)


def get_devices_by_risk(risk_level: DeviceRiskLevel):
    """
    Get all devices with a specific risk level.

    Args:
        risk_level: DeviceRiskLevel enum value

    Returns:
        list: List of device instances with that risk level
    """
    registry = get_registry()
    return registry.get_devices_by_risk(risk_level)


def get_integration_summary():
    """
    Get summary of integrated devices.

    Returns:
        dict: Summary information
    """
    registry = get_registry()
    summary = registry.get_registry_summary()

    summary["version"] = VERSION
    summary["integration_name"] = INTEGRATION_NAME

    return summary


def list_devices():
    """
    List all registered devices with details.

    Returns:
        list: List of device information dictionaries
    """
    registry = get_registry()
    return registry.list_devices()


def print_integration_summary():
    """Print formatted integration summary"""
    print("=" * 80)
    print(f"{INTEGRATION_NAME} v{VERSION}")
    print("=" * 80)

    summary = get_integration_summary()

    print(f"\nTotal Registered Devices: {summary['total_registered']}")
    print(f"Enabled: {summary['enabled']}")
    print(f"Initialized: {summary['initialized']}")

    print("\nBy Group:")
    for group_name, count in sorted(summary['by_group'].items()):
        print(f"  {group_name:40} {count:3} device(s)")

    print("\nBy Risk Level:")
    for risk_name, count in sorted(summary['by_risk'].items()):
        print(f"  {risk_name:15} {count:3} device(s)")

    print("\n" + "=" * 80)


def print_device_list():
    """Print formatted device list"""
    devices = list_devices()

    print("\n" + "=" * 120)
    print("DSMIL Integrated Devices")
    print("=" * 120)
    print(f"{'Device ID':12} {'Name':30} {'Group':25} {'Risk':12} {'State':12}")
    print("-" * 120)

    for device_info in devices:
        device_id = device_info['device_id']
        name = device_info['name'].replace('Device', '').strip()[:29]
        group = device_info['group'][:24]
        risk = device_info['risk_level'][:11]
        state = device_info.get('state', 'uninitialized')[:11]

        print(f"{device_id:12} {name:30} {group:25} {risk:12} {state:12}")

    print("=" * 120)


# Auto-registration on import
register_all_devices()


if __name__ == "__main__":
    """Command-line interface for device integration"""
    import argparse

    parser = argparse.ArgumentParser(
        description="DSMIL Device Integration Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show integration summary
  python3 dsmil_integration.py --summary

  # List all devices
  python3 dsmil_integration.py --list

  # Initialize all devices
  python3 dsmil_integration.py --initialize

  # Get device info
  python3 dsmil_integration.py --device 0x8000

  # Test a specific device
  python3 dsmil_integration.py --test 0x8000
        """
    )

    parser.add_argument('--summary', action='store_true',
                       help='Show integration summary')
    parser.add_argument('--list', action='store_true',
                       help='List all registered devices')
    parser.add_argument('--initialize', action='store_true',
                       help='Initialize all devices')
    parser.add_argument('--device', type=lambda x: int(x, 0),
                       help='Get info for specific device (e.g., 0x8000)')
    parser.add_argument('--test', type=lambda x: int(x, 0),
                       help='Test a specific device (e.g., 0x8000)')

    args = parser.parse_args()

    if args.summary or not any(vars(args).values()):
        print_integration_summary()

    if args.list:
        print_device_list()

    if args.initialize:
        print("\nInitializing all devices...")
        results = initialize_all_devices()

        print(f"\nInitialization Results:")
        for device_id, result in results.items():
            status = "SUCCESS" if result.success else "FAILED"
            print(f"  0x{device_id:04X}: {status}")
            if not result.success:
                print(f"    Error: {result.error}")

    if args.device:
        device = get_device(args.device)
        if device:
            print(f"\nDevice 0x{args.device:04X}: {device.name}")
            print(f"  Description: {device.description}")
            print(f"  State: {device.state.value}")
            print(f"  Capabilities: {len(device.get_capabilities())}")
            print(f"  Registers: {len(device.get_register_map())}")

            status = device.get_status()
            print(f"\n  Status:")
            for key, value in status.items():
                print(f"    {key}: {value}")
        else:
            print(f"Device 0x{args.device:04X} not found or not registered")

    if args.test:
        device = get_device(args.test)
        if device:
            print(f"\nTesting device 0x{args.test:04X}: {device.name}")

            # Initialize
            print("  Initializing...")
            result = device.initialize()
            if result.success:
                print(f"    SUCCESS: {result.data}")
            else:
                print(f"    FAILED: {result.error}")

            # Read registers
            print("  Reading registers...")
            for reg_name in list(device.get_register_map().keys())[:3]:
                result = device.read_register(reg_name)
                if result.success:
                    print(f"    {reg_name}: {result.data.get('hex', 'N/A')}")
                else:
                    print(f"    {reg_name}: ERROR - {result.error}")

            # Get status
            print("  Getting status...")
            status = device.get_status()
            print(f"    {status}")

        else:
            print(f"Device 0x{args.test:04X} not found or not registered")
