#!/usr/bin/env python3
"""
⚠️  DEPRECATED - This component will be removed in v3.0.0 (2026 Q3)
⚠️  Use: DSMILIntegrationAdapter.discover_all_devices_cascading()
⚠️  See DEPRECATION_PLAN.md for migration guide

DSMIL Auto-Discovery Module

Automatically discovers and registers ALL DSMIL devices by scanning
the devices directory. This eliminates manual device registration.

Author: DSMIL Integration Framework
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
"""

import sys
import os
import importlib
import glob
import re
from pathlib import Path

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'devices'))

from lib.device_registry import (
    DSMILDeviceRegistry, get_registry, DeviceGroup, DeviceRiskLevel
)


# ============================================================================
# DEPRECATION WARNING - REMOVE IN v3.0.0 (2026 Q3)
# ============================================================================
import warnings
warnings.warn(
    "\n" + "=" * 80 + "\n"
    "⚠️  DEPRECATED: dsmil_auto_discover.py\n\n"
    "This component is deprecated and will be removed in v3.0.0 (2026 Q3).\n\n"
    "Migration:\n"
    "  from dsmil_integration_adapter import quick_discover\n"
    "  adapter = quick_discover()\n\n"
    "See DEPRECATION_PLAN.md for complete migration guide.\n"
    + "=" * 80,
    DeprecationWarning,
    stacklevel=2
)
# ============================================================================


# Quarantined devices (NEVER load)
QUARANTINED_DEVICES = [
    0x8009,  # Self-Destruct
    0x800A,  # Secure Erase
    0x800B,  # Emergency Lockdown
    0x8019,  # Remote Disable
    0x8029,  # System Reset
]


def discover_device_files():
    """Discover all device files in the devices directory"""
    devices_dir = os.path.join(os.path.dirname(__file__), 'devices')
    pattern = os.path.join(devices_dir, 'device_0x*.py')

    device_files = []
    for filepath in glob.glob(pattern):
        filename = os.path.basename(filepath)

        # Extract device ID from filename
        match = re.match(r'device_(0x[0-9A-Fa-f]+)_.*\.py$', filename)
        if match:
            device_id_str = match.group(1)
            device_id = int(device_id_str, 16)

            # Skip quarantined devices
            if device_id in QUARANTINED_DEVICES:
                continue

            device_files.append({
                'filepath': filepath,
                'filename': filename,
                'device_id': device_id,
                'module_name': filename[:-3],  # Remove .py
            })

    # Sort by device ID
    device_files.sort(key=lambda x: x['device_id'])
    return device_files


def get_device_group(device_id):
    """Determine device group based on device ID"""
    device_num = device_id - 0x8000
    group_num = device_num // 12

    group_map = {
        0: DeviceGroup.GROUP_0_CORE_SECURITY,
        1: DeviceGroup.GROUP_1_EXTENDED_SECURITY,
        2: DeviceGroup.GROUP_2_NETWORK_COMMS,
        3: DeviceGroup.GROUP_3_DATA_PROCESSING,
        4: DeviceGroup.GROUP_4_STORAGE_MANAGEMENT,
        5: DeviceGroup.GROUP_5_PERIPHERAL_CONTROL,
        6: DeviceGroup.GROUP_6_TRAINING_SIMULATION,
    }

    return group_map.get(group_num, DeviceGroup.GROUP_0_CORE_SECURITY)


def get_device_risk_level(device_id):
    """Determine risk level based on device ID"""
    # Core security devices are monitored
    if 0x8000 <= device_id <= 0x8008:
        return DeviceRiskLevel.MONITORED

    # Most other devices are safe
    return DeviceRiskLevel.SAFE


def load_device_class(module_name):
    """Dynamically load device class from module"""
    try:
        module = importlib.import_module(module_name)

        # Find the device class (should end with 'Device')
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and
                attr_name.endswith('Device') and
                attr_name != 'DSMILDeviceBase'):
                return attr

        return None
    except Exception as e:
        print(f"Error loading {module_name}: {e}")
        return None


def auto_register_all_devices():
    """
    Automatically discover and register all DSMIL devices.

    Scans the devices directory, dynamically imports device classes,
    and registers them with the global registry.

    Returns:
        tuple: (registry, stats_dict)
    """
    registry = get_registry()
    stats = {
        'discovered': 0,
        'loaded': 0,
        'registered': 0,
        'failed': 0,
        'quarantined': len(QUARANTINED_DEVICES),
    }

    device_files = discover_device_files()
    stats['discovered'] = len(device_files)

    for device_info in device_files:
        device_id = device_info['device_id']
        module_name = device_info['module_name']

        # Load device class
        device_class = load_device_class(module_name)
        if not device_class:
            stats['failed'] += 1
            continue

        stats['loaded'] += 1

        # Register device
        try:
            registry.register_device(
                device_id=device_id,
                device_class=device_class,
                risk_level=get_device_risk_level(device_id),
                group=get_device_group(device_id),
                enabled=True
            )
            stats['registered'] += 1
        except Exception as e:
            print(f"Failed to register {device_id:#06x}: {e}")
            stats['failed'] += 1

    return registry, stats


# Auto-register on import
registry, stats = auto_register_all_devices()

# Export for compatibility
def get_device(device_id):
    """Get a device by ID"""
    return registry.get_device(device_id)


def get_all_devices():
    """Get all devices"""
    return registry.get_all_devices()


def get_devices_by_group(group):
    """Get devices in a specific group"""
    return registry.get_devices_by_group(group)


def initialize_all_devices():
    """Initialize all devices"""
    results = {}
    for device_id, device in registry.get_all_devices().items():
        results[device_id] = device.initialize()
    return results


def list_devices():
    """List all registered devices with metadata"""
    return registry.list_devices()


def get_integration_summary():
    """Get integration summary"""
    devices = registry.get_all_devices()
    reg_summary = registry.get_registry_summary()

    # Count initialized
    initialized = sum(1 for d in devices.values() if d.state.value == 'ready')

    return {
        'integration_name': "DSMIL Auto-Discovery Integration",
        'version': "2.0.0",
        'total_registered': reg_summary['total_registered'],
        'enabled': reg_summary['enabled'],
        'initialized': initialized,
        'quarantined': len(QUARANTINED_DEVICES),
        'by_group': reg_summary['by_group'],
        'by_risk': reg_summary['by_risk'],
        'auto_discovery_stats': stats,
    }


if __name__ == "__main__":
    summary = get_integration_summary()
    print("DSMIL Auto-Discovery Integration")
    print("=" * 60)
    print(f"Total Registered: {summary['total_registered']}")
    print(f"Quarantined: {summary['quarantined']}")
    print(f"\nDiscovery Stats:")
    print(f"  Discovered: {stats['discovered']} device files")
    print(f"  Loaded: {stats['loaded']} device classes")
    print(f"  Registered: {stats['registered']} devices")
    print(f"  Failed: {stats['failed']}")
    print(f"\nBy Group:")
    for group, count in sorted(summary['by_group'].items()):
        print(f"  {group}: {count}")
