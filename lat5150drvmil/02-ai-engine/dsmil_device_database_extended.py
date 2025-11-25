#!/usr/bin/env python3
"""
DSMIL Extended Device Database - 104 Core + Expansion Devices
==============================================================
Extended database for dsmil-104dev.c driver v5.2.0 with expansion slots

Core Devices (104 devices per driver spec):
- Group 0 (0x8000-0x800B): Core Security & Emergency (12 devices)
- Group 1 (0x8010-0x801B): Extended Security (12 devices)
- Group 2 (0x8020-0x802B): Network & Communications (12 devices)
- Group 3 (0x8030-0x803B): Data Processing (12 devices)
- Group 4 (0x8040-0x804B): Storage Control (12 devices)
- Group 5 (0x8050-0x805B): Peripheral Management (12 devices)
- Group 6 (0x8060-0x806B): Training Functions (12 devices)
- Group 7 (0x8070-0x807B): Diagnostic Tools (12 devices)
- Group 8 (0x8080-0x8087): Extended Capabilities (8 devices)

Expansion Devices (not in driver spec, may be discovered):
- Group 8 Extended (0x8088-0x808B): Additional capabilities (4 devices)
- Extended Slots (0x808C-0x809F): Reserved expansion (20 devices)

Total Database: 128 device slots (104 spec + 24 expansion)
Token Range: 0x8000-0x80FF (256 slots for 104 devices × 3 tokens each)
Device ID to Token: Base_Token = 0x8000 + (device_id × 3)
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Tuple

# Import base definitions from original database
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from dsmil_device_database import (
        DeviceStatus, DeviceGroup, DSMILDevice,
        GROUP_0_DEVICES, GROUP_1_DEVICES, GROUP_2_DEVICES,
        GROUP_3_DEVICES, GROUP_4_DEVICES, GROUP_5_DEVICES,
        GROUP_6_DEVICES, QUARANTINED_DEVICES, SAFE_DEVICES
    )
    LEGACY_IMPORT = True
except ImportError:
    LEGACY_IMPORT = False
    # Define basic types if import fails
    class DeviceStatus(Enum):
        SAFE = "safe"
        QUARANTINED = "quarantined"
        RISKY = "risky"
        UNKNOWN = "unknown"

    class DeviceGroup(Enum):
        GROUP_0_CORE_SECURITY = 0
        GROUP_1_EXTENDED_SECURITY = 1
        GROUP_2_NETWORK_COMM = 2
        GROUP_3_DATA_PROCESSING = 3
        GROUP_4_STORAGE_CONTROL = 4
        GROUP_5_PERIPHERAL_MGT = 5
        GROUP_6_TRAINING = 6
        GROUP_7_DIAGNOSTIC = 7
        GROUP_8_ADVANCED = 8

    @dataclass
    class DSMILDevice:
        device_id: int
        name: str
        status: DeviceStatus
        description: str
        safe_to_activate: bool
        group: DeviceGroup
        read_safe: bool = False
        write_safe: bool = False
        monitored: bool = False


# Extend DeviceGroup enum for new groups
if LEGACY_IMPORT:
    DeviceGroup.GROUP_7_DIAGNOSTIC = 7
    DeviceGroup.GROUP_8_ADVANCED = 8


# ============================================================================
# GROUP 7: Diagnostic Tools (0x8070-0x807B) - NEW
# ============================================================================

GROUP_7_DEVICES = {
    0x8070: DSMILDevice(
        0x8070, "System Diagnostics", DeviceStatus.SAFE,
        "Comprehensive system diagnostic controller",
        True, DeviceGroup.GROUP_7_DIAGNOSTIC,
        read_safe=True, write_safe=True, monitored=False
    ),
    0x8071: DSMILDevice(
        0x8071, "Memory Tester", DeviceStatus.SAFE,
        "RAM diagnostics and testing",
        True, DeviceGroup.GROUP_7_DIAGNOSTIC,
        read_safe=True, write_safe=True, monitored=False
    ),
    0x8072: DSMILDevice(
        0x8072, "CPU Diagnostics", DeviceStatus.SAFE,
        "CPU health and performance testing",
        True, DeviceGroup.GROUP_7_DIAGNOSTIC,
        read_safe=True, write_safe=True, monitored=False
    ),
    0x8073: DSMILDevice(
        0x8073, "Storage Diagnostics", DeviceStatus.SAFE,
        "Storage device health monitoring",
        True, DeviceGroup.GROUP_7_DIAGNOSTIC,
        read_safe=True, write_safe=True, monitored=False
    ),
    0x8074: DSMILDevice(
        0x8074, "Network Diagnostics", DeviceStatus.SAFE,
        "Network connectivity testing",
        True, DeviceGroup.GROUP_7_DIAGNOSTIC,
        read_safe=True, write_safe=True, monitored=False
    ),
    0x8075: DSMILDevice(
        0x8075, "Thermal Diagnostics", DeviceStatus.SAFE,
        "Thermal system testing and calibration",
        True, DeviceGroup.GROUP_7_DIAGNOSTIC,
        read_safe=True, write_safe=True, monitored=False
    ),
    0x8076: DSMILDevice(
        0x8076, "Power Diagnostics", DeviceStatus.SAFE,
        "Power subsystem testing",
        True, DeviceGroup.GROUP_7_DIAGNOSTIC,
        read_safe=True, write_safe=True, monitored=False
    ),
    0x8077: DSMILDevice(
        0x8077, "I/O Diagnostics", DeviceStatus.SAFE,
        "Input/output subsystem testing",
        True, DeviceGroup.GROUP_7_DIAGNOSTIC,
        read_safe=True, write_safe=True, monitored=False
    ),
    0x8078: DSMILDevice(
        0x8078, "Sensor Diagnostics", DeviceStatus.SAFE,
        "Sensor calibration and testing",
        True, DeviceGroup.GROUP_7_DIAGNOSTIC,
        read_safe=True, write_safe=True, monitored=False
    ),
    0x8079: DSMILDevice(
        0x8079, "Firmware Diagnostics", DeviceStatus.SAFE,
        "Firmware integrity verification",
        True, DeviceGroup.GROUP_7_DIAGNOSTIC,
        read_safe=True, write_safe=True, monitored=False
    ),
    0x807A: DSMILDevice(
        0x807A, "Security Diagnostics", DeviceStatus.SAFE,
        "Security subsystem testing",
        True, DeviceGroup.GROUP_7_DIAGNOSTIC,
        read_safe=True, write_safe=True, monitored=False
    ),
    0x807B: DSMILDevice(
        0x807B, "Comprehensive Test", DeviceStatus.SAFE,
        "Full system diagnostic suite",
        True, DeviceGroup.GROUP_7_DIAGNOSTIC,
        read_safe=True, write_safe=True, monitored=False
    ),
}


# ============================================================================
# GROUP 8: Advanced Features (0x8080-0x808B) - NEW
# ============================================================================

GROUP_8_DEVICES = {
    0x8080: DSMILDevice(
        0x8080, "Machine Learning Accelerator", DeviceStatus.UNKNOWN,
        "Hardware ML acceleration interface",
        False, DeviceGroup.GROUP_8_ADVANCED,
        read_safe=True, write_safe=False
    ),
    0x8081: DSMILDevice(
        0x8081, "Quantum RNG", DeviceStatus.UNKNOWN,
        "Quantum random number generator",
        False, DeviceGroup.GROUP_8_ADVANCED,
        read_safe=True, write_safe=False
    ),
    0x8082: DSMILDevice(
        0x8082, "Neural Processor", DeviceStatus.UNKNOWN,
        "Neural processing unit interface",
        False, DeviceGroup.GROUP_8_ADVANCED,
        read_safe=True, write_safe=False
    ),
    0x8083: DSMILDevice(
        0x8083, "Cryptographic Accelerator", DeviceStatus.UNKNOWN,
        "Hardware crypto acceleration",
        False, DeviceGroup.GROUP_8_ADVANCED,
        read_safe=True, write_safe=False
    ),
    0x8084: DSMILDevice(
        0x8084, "Signal Processing", DeviceStatus.UNKNOWN,
        "Digital signal processing unit",
        False, DeviceGroup.GROUP_8_ADVANCED,
        read_safe=True, write_safe=False
    ),
    0x8085: DSMILDevice(
        0x8085, "Vector Processing", DeviceStatus.UNKNOWN,
        "SIMD vector processing unit",
        False, DeviceGroup.GROUP_8_ADVANCED,
        read_safe=True, write_safe=False
    ),
    0x8086: DSMILDevice(
        0x8086, "Compression Accelerator", DeviceStatus.UNKNOWN,
        "Hardware compression/decompression",
        False, DeviceGroup.GROUP_8_ADVANCED,
        read_safe=True, write_safe=False
    ),
    0x8087: DSMILDevice(
        0x8087, "Pattern Recognition", DeviceStatus.UNKNOWN,
        "Hardware pattern matching engine",
        False, DeviceGroup.GROUP_8_ADVANCED,
        read_safe=True, write_safe=False
    ),
    # Extended Group 8 devices (0x8088-0x808B) - Not in driver spec but may be discovered
    0x8088: DSMILDevice(
        0x8088, "Image Processing", DeviceStatus.UNKNOWN,
        "Hardware image processing unit (undiscovered/experimental)",
        False, DeviceGroup.GROUP_8_ADVANCED,
        read_safe=True, write_safe=False
    ),
    0x8089: DSMILDevice(
        0x8089, "Video Accelerator", DeviceStatus.UNKNOWN,
        "Video encoding/decoding hardware (undiscovered/experimental)",
        False, DeviceGroup.GROUP_8_ADVANCED,
        read_safe=True, write_safe=False
    ),
    0x808A: DSMILDevice(
        0x808A, "Audio DSP", DeviceStatus.UNKNOWN,
        "Audio digital signal processor (undiscovered/experimental)",
        False, DeviceGroup.GROUP_8_ADVANCED,
        read_safe=True, write_safe=False
    ),
    0x808B: DSMILDevice(
        0x808B, "Coprocessor Interface", DeviceStatus.UNKNOWN,
        "General-purpose coprocessor interface (undiscovered/experimental)",
        False, DeviceGroup.GROUP_8_ADVANCED,
        read_safe=True, write_safe=False
    ),
}


# ============================================================================
# EXTENDED DEVICES (0x808C-0x809F) - Expansion slots for future discovery
# ============================================================================
# Note: These devices are not in the driver spec (v5.2.0) but slots are reserved
# for potential future discovery of additional hardware capabilities

EXTENDED_DEVICES = {}
for i in range(20):
    device_id = 0x808C + i
    EXTENDED_DEVICES[device_id] = DSMILDevice(
        device_id, f"Extended Device {i+1}", DeviceStatus.UNKNOWN,
        f"Reserved expansion slot {i+1} (may be discovered)",
        False, DeviceGroup.GROUP_8_ADVANCED,
        read_safe=True, write_safe=False
    )


# ============================================================================
# Complete 104-Device Database
# ============================================================================

ALL_DEVICES_EXTENDED = {}

# Add legacy devices if available
if LEGACY_IMPORT:
    ALL_DEVICES_EXTENDED.update(GROUP_0_DEVICES)
    ALL_DEVICES_EXTENDED.update(GROUP_1_DEVICES)
    ALL_DEVICES_EXTENDED.update(GROUP_2_DEVICES)
    ALL_DEVICES_EXTENDED.update(GROUP_3_DEVICES)
    ALL_DEVICES_EXTENDED.update(GROUP_4_DEVICES)
    ALL_DEVICES_EXTENDED.update(GROUP_5_DEVICES)
    ALL_DEVICES_EXTENDED.update(GROUP_6_DEVICES)

# Add new groups
ALL_DEVICES_EXTENDED.update(GROUP_7_DEVICES)
ALL_DEVICES_EXTENDED.update(GROUP_8_DEVICES)
ALL_DEVICES_EXTENDED.update(EXTENDED_DEVICES)


# Quarantined devices (from legacy + new)
QUARANTINED_DEVICES_EXTENDED = QUARANTINED_DEVICES if LEGACY_IMPORT else [
    0x8009, 0x800A, 0x800B, 0x8019, 0x8029
]

# Safe devices (from legacy + new diagnostic group)
SAFE_DEVICES_EXTENDED = (SAFE_DEVICES if LEGACY_IMPORT else [
    0x8003, 0x8004, 0x8005, 0x8006, 0x8007, 0x802A
]) + list(range(0x8070, 0x807C))  # All Group 7 diagnostic devices are safe


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_device_extended(device_id: int) -> DSMILDevice:
    """Get device by ID"""
    return ALL_DEVICES_EXTENDED.get(device_id)


def get_devices_by_group_extended(group: DeviceGroup) -> Dict[int, DSMILDevice]:
    """Get all devices in a group"""
    return {k: v for k, v in ALL_DEVICES_EXTENDED.items() if v.group == group}


def get_devices_by_status_extended(status: DeviceStatus) -> Dict[int, DSMILDevice]:
    """Get all devices with a specific status"""
    return {k: v for k, v in ALL_DEVICES_EXTENDED.items() if v.status == status}


def get_safe_devices_extended() -> Dict[int, DSMILDevice]:
    """Get all safe devices"""
    return get_devices_by_status_extended(DeviceStatus.SAFE)


def get_quarantined_devices_extended() -> Dict[int, DSMILDevice]:
    """Get all quarantined devices"""
    return get_devices_by_status_extended(DeviceStatus.QUARANTINED)


def get_device_by_token(token_id: int) -> DSMILDevice:
    """
    Get device by token ID

    Each device has 3 tokens:
    - Base + 0: Status token
    - Base + 1: Config token
    - Base + 2: Data token
    """
    if token_id < 0x8000 or token_id > 0x8137:
        return None

    # Calculate device ID from token
    token_offset = token_id - 0x8000
    device_id = token_offset // 3

    return get_device_extended(device_id)


def get_token_range(device_id: int) -> Tuple[int, int, int]:
    """
    Get the 3 token IDs for a device

    Returns: (status_token, config_token, data_token)
    """
    base_token = 0x8000 + (device_id * 3)
    return (base_token, base_token + 1, base_token + 2)


def get_statistics_extended() -> Dict:
    """Get extended device database statistics"""
    return {
        "total_devices": len(ALL_DEVICES_EXTENDED),
        "safe": len(get_safe_devices_extended()),
        "quarantined": len(get_quarantined_devices_extended()),
        "risky": len(get_devices_by_status_extended(DeviceStatus.RISKY)),
        "unknown": len(get_devices_by_status_extended(DeviceStatus.UNKNOWN)),
        "groups": {
            "group_0_core_security": len(get_devices_by_group_extended(DeviceGroup.GROUP_0_CORE_SECURITY)) if LEGACY_IMPORT else 0,
            "group_1_extended_security": len(get_devices_by_group_extended(DeviceGroup.GROUP_1_EXTENDED_SECURITY)) if LEGACY_IMPORT else 0,
            "group_2_network": len(get_devices_by_group_extended(DeviceGroup.GROUP_2_NETWORK_COMM)) if LEGACY_IMPORT else 0,
            "group_3_data_processing": len(get_devices_by_group_extended(DeviceGroup.GROUP_3_DATA_PROCESSING)) if LEGACY_IMPORT else 0,
            "group_4_storage": len(get_devices_by_group_extended(DeviceGroup.GROUP_4_STORAGE_CONTROL)) if LEGACY_IMPORT else 0,
            "group_5_peripheral": len(get_devices_by_group_extended(DeviceGroup.GROUP_5_PERIPHERAL_MGT)) if LEGACY_IMPORT else 0,
            "group_6_training": len(get_devices_by_group_extended(DeviceGroup.GROUP_6_TRAINING)) if LEGACY_IMPORT else 0,
            "group_7_diagnostic": len(GROUP_7_DEVICES),
            "group_8_advanced": len(GROUP_8_DEVICES),
            "extended": len(EXTENDED_DEVICES),
        },
        "token_range": {
            "start": 0x8000,
            "end": 0x8137,
            "count": (0x8137 - 0x8000 + 1) // 3,  # 104 devices
        }
    }


if __name__ == "__main__":
    """Print extended database statistics"""
    stats = get_statistics_extended()
    print("="*70)
    print(" DSMIL EXTENDED DEVICE DATABASE (104 DEVICES)")
    print("="*70)
    print(f"\nTotal Devices: {stats['total_devices']}")
    print(f"Safe:          {stats['safe']} ({stats['safe']/stats['total_devices']*100:.1f}%)")
    print(f"Quarantined:   {stats['quarantined']} ({stats['quarantined']/stats['total_devices']*100:.1f}%)")
    print(f"Risky:         {stats['risky']} ({stats['risky']/stats['total_devices']*100:.1f}%)")
    print(f"Unknown:       {stats['unknown']} ({stats['unknown']/stats['total_devices']*100:.1f}%)")

    print("\nDevices by Group:")
    for group_name, count in stats['groups'].items():
        if count > 0:
            print(f"  {group_name}: {count} devices")

    print(f"\nToken Range: 0x{stats['token_range']['start']:04X}-0x{stats['token_range']['end']:04X}")
    print(f"Token Count: {stats['token_range']['count']} devices × 3 tokens = {stats['token_range']['count']*3} tokens")

    print("\n" + "="*70)
    print(f" {stats['total_devices']} TOTAL DEVICES (104 device architecture)")
    print(f" {stats['safe']} SAFE | {stats['quarantined']} QUARANTINED")
    print("="*70)

    # Show new groups
    print("\nNEW DEVICE GROUPS:")
    print("\nGroup 7 - Diagnostic Tools (12 devices):")
    for device_id, device in GROUP_7_DEVICES.items():
        print(f"  0x{device_id:04X}: {device.name}")

    print("\nGroup 8 - Advanced Features (12 devices):")
    for device_id, device in list(GROUP_8_DEVICES.items())[:5]:
        print(f"  0x{device_id:04X}: {device.name}")
    print(f"  ... and {len(GROUP_8_DEVICES) - 5} more")

    print(f"\nExtended Devices: {len(EXTENDED_DEVICES)} expansion slots")
