#!/usr/bin/env python3
"""
DSMIL Device Base Class

Base class for all DSMIL device integrations providing common interface
and safety mechanisms.

Author: DSMIL Integration Framework
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import time


class DeviceCapability(Enum):
    """Standard device capabilities"""
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"
    CONFIGURATION = "configuration"
    STATUS_REPORTING = "status_reporting"
    EVENT_MONITORING = "event_monitoring"
    INTERRUPT_DRIVEN = "interrupt_driven"
    DMA_CAPABLE = "dma_capable"
    ENCRYPTED_STORAGE = "encrypted_storage"
    MSR_ACCESS = "msr_access"              # Model-Specific Register access
    SMM_ACCESS = "smm_access"              # System Management Mode access
    MEMORY_MAPPED_IO = "memory_mapped_io"  # MMIO access
    FIRMWARE_LEVEL = "firmware_level"      # Firmware/UEFI level operations


class DeviceState(Enum):
    """Device operational states"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    DISABLED = "disabled"
    SUSPENDED = "suspended"


class OperationResult:
    """Result of a device operation"""

    def __init__(self, success: bool, data: Any = None, error: str = None,
                 duration: float = None):
        self.success = success
        self.data = data
        self.error = error
        self.duration = duration
        self.timestamp = time.time()

    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "duration_ms": round(self.duration * 1000, 2) if self.duration else None,
            "timestamp": self.timestamp,
        }


class DSMILDeviceBase(ABC):
    """Base class for DSMIL device implementations"""

    def __init__(self, device_id: int, name: str, description: str):
        self.device_id = device_id
        self.name = name
        self.description = description
        self.state = DeviceState.UNINITIALIZED
        self.capabilities = []
        self.register_map = {}
        self.operation_count = 0
        self.error_count = 0
        self.last_error = None

    @abstractmethod
    def initialize(self) -> OperationResult:
        """
        Initialize the device

        Returns:
            OperationResult with success status
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> List[DeviceCapability]:
        """
        Get device capabilities

        Returns:
            List of DeviceCapability values
        """
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """
        Get current device status

        Returns:
            Dictionary with device status information
        """
        pass

    @abstractmethod
    def read_register(self, register: str) -> OperationResult:
        """
        Read a device register

        Args:
            register: Register name or address

        Returns:
            OperationResult with register value
        """
        pass

    def write_register(self, register: str, value: int) -> OperationResult:
        """
        Write to a device register (default: not supported)

        Args:
            register: Register name or address
            value: Value to write

        Returns:
            OperationResult with success status
        """
        return OperationResult(False, error="Write operations not supported")

    def reset(self) -> OperationResult:
        """
        Reset the device (default: not supported)

        Returns:
            OperationResult with success status
        """
        return OperationResult(False, error="Reset operation not supported")

    def get_register_map(self) -> Dict[str, Dict]:
        """
        Get the device register map

        Returns:
            Dictionary mapping register names to their properties
        """
        return self.register_map

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get device operation statistics

        Returns:
            Dictionary with statistics
        """
        return {
            "device_id": f"0x{self.device_id:04X}",
            "name": self.name,
            "state": self.state.value,
            "operation_count": self.operation_count,
            "error_count": self.error_count,
            "last_error": self.last_error,
        }

    def _record_operation(self, success: bool, error: str = None):
        """Record an operation for statistics"""
        self.operation_count += 1
        if not success:
            self.error_count += 1
            self.last_error = error

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} 0x{self.device_id:04X} '{self.name}' state={self.state.value}>"
