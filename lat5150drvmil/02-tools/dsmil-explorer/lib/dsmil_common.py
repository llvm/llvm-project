#!/usr/bin/env python3
"""
DSMIL Common Utilities Library

Shared utilities for DSMIL device operations, including device I/O,
register access, and common helper functions.

Author: DSMIL Automation Framework
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
"""

import os
import sys
import struct
import mmap
import fcntl
from typing import Optional, List, Tuple
from enum import Enum

# DSMIL Hardware Constants
SMI_COMMAND_PORT = 0xB2
SMI_STATUS_PORT = 0xB3
DELL_LEGACY_IO_BASE = 0x164E
DELL_LEGACY_IO_DATA = 0x164F

# Device registry base address
DEVICE_REGISTRY_BASE = 0x60000000
DEVICE_REGISTRY_SIZE = 0x1000  # 4KB per device
TOTAL_DEVICES = 84

class DSMILDevice:
    """Represents a DSMIL hardware device"""

    def __init__(self, device_id: int, name: str = None):
        if device_id < 0x8000 or device_id >= 0x8000 + TOTAL_DEVICES:
            raise ValueError(f"Invalid device ID: 0x{device_id:04X}")

        self.device_id = device_id
        self.device_index = device_id - 0x8000
        self.name = name or f"Device_0x{device_id:04X}"
        self.base_address = DEVICE_REGISTRY_BASE + (self.device_index * DEVICE_REGISTRY_SIZE)

    def __str__(self):
        return f"DSMIL Device 0x{self.device_id:04X} ({self.name}) @ 0x{self.base_address:08X}"

    def __repr__(self):
        return f"DSMILDevice(0x{self.device_id:04X}, '{self.name}')"

class DeviceAccess:
    """
    Low-level device access via kernel module

    Provides safe access to DSMIL devices through the kernel driver
    """

    DSMIL_DEVICE_PATH = "/dev/dsmil0"
    IOCTL_MAGIC = 0xD5

    # IOCTL commands
    IOCTL_READ_REGISTER = (IOCTL_MAGIC << 8) | 0x01
    IOCTL_WRITE_REGISTER = (IOCTL_MAGIC << 8) | 0x02
    IOCTL_GET_STATUS = (IOCTL_MAGIC << 8) | 0x03
    IOCTL_GET_CAPABILITIES = (IOCTL_MAGIC << 8) | 0x04

    def __init__(self):
        self.dev_fd = None
        self.is_open = False

    def open(self) -> bool:
        """Open DSMIL device"""
        try:
            if not os.path.exists(self.DSMIL_DEVICE_PATH):
                print(f"Error: DSMIL device not found at {self.DSMIL_DEVICE_PATH}")
                print("Hint: Load kernel module with: sudo insmod dsmil-72dev.ko")
                return False

            self.dev_fd = os.open(self.DSMIL_DEVICE_PATH, os.O_RDWR)
            self.is_open = True
            return True

        except PermissionError:
            print(f"Error: Permission denied accessing {self.DSMIL_DEVICE_PATH}")
            print("Hint: Run with sudo or add user to dsmil group")
            return False
        except Exception as e:
            print(f"Error opening device: {e}")
            return False

    def close(self):
        """Close DSMIL device"""
        if self.dev_fd is not None:
            os.close(self.dev_fd)
            self.dev_fd = None
            self.is_open = False

    def read_register(self, device_id: int, offset: int, size: int = 4) -> Optional[int]:
        """
        Read device register

        Args:
            device_id: Device ID (0x8000-0x806B)
            offset: Register offset
            size: Register size (1, 2, 4, or 8 bytes)

        Returns:
            Register value or None on error
        """
        if not self.is_open:
            print("Error: Device not open")
            return None

        try:
            # Pack request: device_id (2 bytes), offset (4 bytes), size (1 byte)
            request = struct.pack("<HI B", device_id, offset, size)

            # Send IOCTL
            result = fcntl.ioctl(self.dev_fd, self.IOCTL_READ_REGISTER, request)

            # Unpack result based on size
            if size == 1:
                value = struct.unpack("<B", result[:1])[0]
            elif size == 2:
                value = struct.unpack("<H", result[:2])[0]
            elif size == 4:
                value = struct.unpack("<I", result[:4])[0]
            elif size == 8:
                value = struct.unpack("<Q", result[:8])[0]
            else:
                return None

            return value

        except Exception as e:
            print(f"Error reading register: {e}")
            return None

    def write_register(self, device_id: int, offset: int, value: int, size: int = 4) -> bool:
        """
        Write device register

        Args:
            device_id: Device ID (0x8000-0x806B)
            offset: Register offset
            value: Value to write
            size: Register size (1, 2, 4, or 8 bytes)

        Returns:
            True on success, False on error
        """
        if not self.is_open:
            print("Error: Device not open")
            return False

        try:
            # Pack request based on size
            if size == 1:
                packed_value = struct.pack("<B", value)
            elif size == 2:
                packed_value = struct.pack("<H", value)
            elif size == 4:
                packed_value = struct.pack("<I", value)
            elif size == 8:
                packed_value = struct.pack("<Q", value)
            else:
                return False

            # Pack full request: device_id (2 bytes), offset (4 bytes), size (1 byte), value
            request = struct.pack("<HI B", device_id, offset, size) + packed_value

            # Send IOCTL
            fcntl.ioctl(self.dev_fd, self.IOCTL_WRITE_REGISTER, request)
            return True

        except Exception as e:
            print(f"Error writing register: {e}")
            return False

    def get_device_status(self, device_id: int) -> Optional[Dict]:
        """
        Get device status

        Returns:
            Dictionary with status information or None
        """
        if not self.is_open:
            return None

        try:
            request = struct.pack("<H", device_id)
            result = fcntl.ioctl(self.dev_fd, self.IOCTL_GET_STATUS, request)

            # Unpack status: state (1 byte), flags (4 bytes), timestamp (8 bytes)
            state, flags, timestamp = struct.unpack("<BIQ", result[:13])

            return {
                "device_id": device_id,
                "state": state,
                "flags": flags,
                "timestamp": timestamp,
                "enabled": bool(flags & 0x01),
                "ready": bool(flags & 0x02),
                "error": bool(flags & 0x04),
            }

        except Exception as e:
            print(f"Error getting device status: {e}")
            return None

    def get_device_capabilities(self, device_id: int) -> Optional[Dict]:
        """
        Get device capabilities

        Returns:
            Dictionary with capability information or None
        """
        if not self.is_open:
            return None

        try:
            request = struct.pack("<H", device_id)
            result = fcntl.ioctl(self.dev_fd, self.IOCTL_GET_CAPABILITIES, request)

            # Unpack capabilities: version (4 bytes), features (4 bytes),
            # max_ops (4 bytes), reserved (4 bytes)
            version, features, max_ops, reserved = struct.unpack("<IIII", result[:16])

            return {
                "device_id": device_id,
                "version": version,
                "features": features,
                "max_operations": max_ops,
                "supports_read": bool(features & 0x01),
                "supports_write": bool(features & 0x02),
                "supports_irq": bool(features & 0x04),
                "supports_dma": bool(features & 0x08),
            }

        except Exception as e:
            print(f"Error getting device capabilities: {e}")
            return None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

def format_hex_dump(data: bytes, base_addr: int = 0, bytes_per_line: int = 16) -> str:
    """Format binary data as hex dump"""
    lines = []
    for i in range(0, len(data), bytes_per_line):
        chunk = data[i:i+bytes_per_line]
        addr = base_addr + i

        # Hex values
        hex_values = ' '.join(f'{b:02X}' for b in chunk)
        hex_values = hex_values.ljust(bytes_per_line * 3)

        # ASCII representation
        ascii_repr = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)

        lines.append(f'{addr:08X}  {hex_values}  {ascii_repr}')

    return '\n'.join(lines)

def check_kernel_module_loaded() -> bool:
    """Check if DSMIL kernel module is loaded"""
    try:
        with open('/proc/modules', 'r') as f:
            modules = f.read()
            return 'dsmil' in modules.lower()
    except:
        return False

def get_device_list() -> List[DSMILDevice]:
    """Get list of all DSMIL devices"""
    devices = []
    for i in range(TOTAL_DEVICES):
        device_id = 0x8000 + i
        devices.append(DSMILDevice(device_id))
    return devices

if __name__ == "__main__":
    # Self-test
    print("DSMIL Common Utilities - Self Test")
    print("=" * 80)

    # Test device creation
    print("\nTesting device creation:")
    device = DSMILDevice(0x8003, "Audit Log Controller")
    print(f"  {device}")
    print(f"  Base Address: 0x{device.base_address:08X}")

    # Check kernel module
    print("\nChecking kernel module:")
    loaded = check_kernel_module_loaded()
    print(f"  DSMIL module loaded: {loaded}")

    # Test device access
    print("\nTesting device access:")
    with DeviceAccess() as dev:
        if dev.is_open:
            print("  Device opened successfully")

            # Try to read device status
            status = dev.get_device_status(0x8003)
            if status:
                print(f"  Device status:")
                for key, value in status.items():
                    print(f"    {key}: {value}")
        else:
            print("  Failed to open device (kernel module not loaded or no permission)")

    # List all devices
    print("\nAll DSMIL devices:")
    devices = get_device_list()
    print(f"  Total devices: {len(devices)}")
    print(f"  Range: 0x{devices[0].device_id:04X} - 0x{devices[-1].device_id:04X}")
