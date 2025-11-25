#!/usr/bin/env python3
"""
DSMIL Device Controller
Interfaces with Track A kernel module for device operations
"""

import os
import asyncio
import struct
import fcntl
import logging
import time
import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta

from config import (settings, get_device_info, assess_operation_risk, 
                    is_device_quarantined, is_device_safe_monitored, get_device_access_level)

logger = logging.getLogger(__name__)


@dataclass
class DeviceState:
    """Device state information"""
    device_id: int
    status: str
    is_active: bool
    is_quarantined: bool
    last_response_time_ms: Optional[int] = None
    error_count: int = 0
    success_count: int = 0
    last_accessed: Optional[datetime] = None


@dataclass 
class OperationResult:
    """Device operation result"""
    operation_id: str
    success: bool
    result_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time_ms: int = 0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class KernelModuleInterface:
    """Interface to DSMIL kernel module"""
    
    def __init__(self):
        self.device_path = settings.kernel_module_path
        self.sysfs_path = settings.sysfs_device_path
        self.device_fd = None
        self.is_available = False
        
        # IOCTL command codes (must match kernel module)
        self.DSMIL_IOCTL_READ = 0x40044400
        self.DSMIL_IOCTL_WRITE = 0x40044401
        self.DSMIL_IOCTL_STATUS = 0x40044402
        self.DSMIL_IOCTL_RESET = 0x40044403
        
    async def initialize(self) -> bool:
        """Initialize kernel module interface"""
        try:
            # Check if kernel module is loaded
            if not os.path.exists(self.device_path):
                logger.warning(f"Kernel module device not found: {self.device_path}")
                return False
            
            # Open device file
            try:
                self.device_fd = os.open(self.device_path, os.O_RDWR)
                logger.info(f"Opened kernel module device: {self.device_path}")
            except PermissionError:
                logger.error(f"Permission denied accessing {self.device_path}")
                return False
            except OSError as e:
                logger.error(f"Failed to open kernel module device: {e}")
                return False
            
            # Test basic communication
            test_result = await self.test_communication()
            if not test_result:
                logger.warning("Kernel module communication test failed")
                return False
            
            self.is_available = True
            logger.info("Kernel module interface initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize kernel module interface: {e}")
            return False
    
    async def test_communication(self) -> bool:
        """Test communication with kernel module"""
        try:
            if not self.device_fd:
                return False
            
            # Try to get system status
            status_data = struct.pack('I', 0)  # Request system status
            result = fcntl.ioctl(self.device_fd, self.DSMIL_IOCTL_STATUS, status_data)
            
            logger.debug("Kernel module communication test successful")
            return True
            
        except Exception as e:
            logger.error(f"Kernel module communication test failed: {e}")
            return False
    
    async def read_device(self, device_id: int, register: str = "STATUS") -> Dict[str, Any]:
        """Read from DSMIL device via kernel module"""
        if not self.is_available or not self.device_fd:
            raise RuntimeError("Kernel module interface not available")
        
        try:
            # Prepare IOCTL data structure
            # Format: device_id (4 bytes) + register_name (16 bytes) + padding
            register_bytes = register.encode('ascii')[:15].ljust(16, b'\0')
            ioctl_data = struct.pack('I16s', device_id, register_bytes)
            
            start_time = time.time()
            
            # Execute IOCTL
            result = fcntl.ioctl(self.device_fd, self.DSMIL_IOCTL_READ, ioctl_data)
            
            execution_time = int((time.time() - start_time) * 1000)
            
            # Parse result
            # Format: status (4 bytes) + data_length (4 bytes) + data (variable)
            status, data_length = struct.unpack('II', result[:8])
            raw_data = result[8:8+data_length] if data_length > 0 else b''
            
            return {
                "status": status,
                "data": raw_data.hex() if raw_data else None,
                "register": register,
                "execution_time_ms": execution_time,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to read from device {device_id:04X}: {e}")
            raise RuntimeError(f"Device read failed: {str(e)}")
    
    async def write_device(self, device_id: int, data: bytes, register: str = "DATA") -> Dict[str, Any]:
        """Write to DSMIL device via kernel module"""
        if not self.is_available or not self.device_fd:
            raise RuntimeError("Kernel module interface not available")
        
        try:
            # Prepare IOCTL data structure
            register_bytes = register.encode('ascii')[:15].ljust(16, b'\0')
            data_length = len(data)
            
            # Format: device_id + register_name + data_length + data
            ioctl_data = struct.pack(f'I16sI{data_length}s', device_id, register_bytes, data_length, data)
            
            start_time = time.time()
            
            # Execute IOCTL
            result = fcntl.ioctl(self.device_fd, self.DSMIL_IOCTL_WRITE, ioctl_data)
            
            execution_time = int((time.time() - start_time) * 1000)
            
            # Parse result
            status = struct.unpack('I', result[:4])[0]
            
            return {
                "status": status,
                "bytes_written": data_length,
                "register": register,
                "execution_time_ms": execution_time,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to write to device {device_id:04X}: {e}")
            raise RuntimeError(f"Device write failed: {str(e)}")
    
    async def get_device_status(self, device_id: int) -> Dict[str, Any]:
        """Get device status via kernel module"""
        try:
            return await self.read_device(device_id, "STATUS")
        except Exception as e:
            logger.error(f"Failed to get status for device {device_id:04X}: {e}")
            return {
                "status": 0xFFFFFFFF,  # Error status
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def reset_device(self, device_id: int) -> Dict[str, Any]:
        """Reset DSMIL device via kernel module"""
        if not self.is_available or not self.device_fd:
            raise RuntimeError("Kernel module interface not available")
        
        try:
            # Prepare reset command
            ioctl_data = struct.pack('I', device_id)
            
            start_time = time.time()
            
            # Execute reset IOCTL
            result = fcntl.ioctl(self.device_fd, self.DSMIL_IOCTL_RESET, ioctl_data)
            
            execution_time = int((time.time() - start_time) * 1000)
            
            # Parse result
            status = struct.unpack('I', result[:4])[0]
            
            return {
                "status": status,
                "operation": "RESET",
                "execution_time_ms": execution_time,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to reset device {device_id:04X}: {e}")
            raise RuntimeError(f"Device reset failed: {str(e)}")
    
    def cleanup(self):
        """Cleanup kernel module interface"""
        if self.device_fd:
            try:
                os.close(self.device_fd)
                logger.info("Kernel module device closed")
            except Exception as e:
                logger.error(f"Error closing kernel module device: {e}")
            finally:
                self.device_fd = None
        
        self.is_available = False


class DeviceController:
    """High-level device controller with safety features"""
    
    def __init__(self):
        self.kernel_interface = KernelModuleInterface()
        self.device_states: Dict[int, DeviceState] = {}
        self.performance_metrics = {
            "operations_per_second": 0.0,
            "average_latency_ms": 0.0,
            "error_rate": 0.0,
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0
        }
        
        # Initialize device registry
        self._initialize_device_registry()
        
        # Operation tracking
        self.active_operations = {}
        self.operation_history = []
        self.last_health_check = datetime.utcnow()
        
    def _initialize_device_registry(self):
        """Initialize device registry with all DSMIL devices"""
        for device_id in range(settings.device_base_id, settings.device_base_id + settings.device_count):
            device_info = get_device_info(device_id)
            
            self.device_states[device_id] = DeviceState(
                device_id=device_id,
                status=device_info["status"],
                is_active=device_info["status"] == "ACTIVE",
                is_quarantined=device_info["is_quarantined"]
            )
    
    async def initialize(self):
        """Initialize device controller"""
        logger.info("Initializing DSMIL device controller...")
        
        # Initialize kernel module interface
        kernel_available = await self.kernel_interface.initialize()
        
        if kernel_available:
            logger.info("Kernel module interface available")
            # Perform initial device health check
            await self.health_check_all_devices()
        else:
            logger.warning("Kernel module interface not available - running in simulation mode")
        
        logger.info("Device controller initialization complete")
    
    async def execute_operation(
        self,
        device_id: int,
        operation_type: str,
        operation_data: Optional[Dict[str, Any]] = None,
        auth_token: Optional[str] = None,
        user_context: Any = None
    ) -> Dict[str, Any]:
        """Execute device operation with safety checks"""
        
        operation_id = f"op_{device_id}_{int(time.time() * 1000)}"
        start_time = time.time()
        
        try:
            # Validate device ID
            if device_id not in self.device_states:
                raise ValueError(f"Invalid device ID: {device_id:04X}")
            
            device_state = self.device_states[device_id]
            
            # Enhanced safety checks based on NSA assessment
            access_level = get_device_access_level(device_id)
            
            # CRITICAL: Absolute quarantine enforcement
            if access_level == "NEVER_ACCESS":
                logger.critical(f"BLOCKED: Attempt to access QUARANTINED device 0x{device_id:04X}")
                raise RuntimeError(f"CRITICAL: Device 0x{device_id:04X} is QUARANTINED - DATA DESTRUCTION RISK")
            
            # Phase 1: Safe devices are READ-ONLY
            if access_level == "READ_ONLY" and operation_type.upper() != "READ":
                raise RuntimeError(f"Device 0x{device_id:04X} is READ-ONLY in Phase 1")
            
            # Unknown devices are restricted
            if access_level == "RESTRICTED":
                logger.warning(f"Access to restricted device 0x{device_id:04X} - operation logged")
                # Could add additional approval requirements here
            
            # Log operation start
            self.active_operations[operation_id] = {
                "device_id": device_id,
                "operation_type": operation_type,
                "started_at": datetime.utcnow(),
                "user_id": getattr(user_context, 'user_id', 'unknown')
            }
            
            # Execute operation based on type
            if operation_type.upper() == "READ":
                result = await self._execute_read_operation(device_id, operation_data or {})
            elif operation_type.upper() == "WRITE":
                result = await self._execute_write_operation(device_id, operation_data or {})
            elif operation_type.upper() == "STATUS":
                result = await self._execute_status_operation(device_id)
            elif operation_type.upper() == "RESET":
                result = await self._execute_reset_operation(device_id)
            else:
                raise ValueError(f"Unsupported operation type: {operation_type}")
            
            # Update device state
            execution_time = int((time.time() - start_time) * 1000)
            device_state.last_response_time_ms = execution_time
            device_state.success_count += 1
            device_state.last_accessed = datetime.utcnow()
            
            # Update performance metrics
            self._update_performance_metrics(True, execution_time)
            
            # Remove from active operations
            if operation_id in self.active_operations:
                del self.active_operations[operation_id]
            
            return {
                "operation_id": operation_id,
                "success": True,
                "result_data": result,
                "execution_time_ms": execution_time,
                "timestamp": datetime.utcnow()
            }
            
        except Exception as e:
            # Update error counts
            if device_id in self.device_states:
                self.device_states[device_id].error_count += 1
            
            execution_time = int((time.time() - start_time) * 1000)
            self._update_performance_metrics(False, execution_time)
            
            # Remove from active operations
            if operation_id in self.active_operations:
                del self.active_operations[operation_id]
            
            logger.error(f"Operation {operation_id} failed: {e}")
            
            return {
                "operation_id": operation_id,
                "success": False,
                "error_message": str(e),
                "execution_time_ms": execution_time,
                "timestamp": datetime.utcnow()
            }
    
    async def _execute_read_operation(self, device_id: int, operation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute read operation"""
        register = operation_data.get("register", "STATUS")
        
        if self.kernel_interface.is_available:
            return await self.kernel_interface.read_device(device_id, register)
        else:
            # Simulation mode
            return {
                "status": 0,
                "data": f"simulated_data_for_device_{device_id:04X}",
                "register": register,
                "execution_time_ms": 10,
                "timestamp": datetime.utcnow().isoformat(),
                "simulation": True
            }
    
    async def _execute_write_operation(self, device_id: int, operation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute write operation with enhanced safety checks"""
        # CRITICAL: Double-check quarantine status before ANY write
        if is_device_quarantined(device_id):
            logger.critical(f"WRITE BLOCKED: Device 0x{device_id:04X} is QUARANTINED")
            raise RuntimeError(f"CRITICAL: Cannot write to quarantined device 0x{device_id:04X}")
        
        # Phase 1: No writes to safe monitoring devices
        if is_device_safe_monitored(device_id):
            raise RuntimeError(f"Device 0x{device_id:04X} is READ-ONLY during Phase 1 monitoring")
        
        data = operation_data.get("data", "")
        register = operation_data.get("register", "DATA")
        
        # Convert data to bytes if it's a string
        if isinstance(data, str):
            data_bytes = bytes.fromhex(data) if data else b'\x00'
        else:
            data_bytes = bytes(data)
        
        if self.kernel_interface.is_available:
            return await self.kernel_interface.write_device(device_id, data_bytes, register)
        else:
            # Simulation mode
            return {
                "status": 0,
                "bytes_written": len(data_bytes),
                "register": register,
                "execution_time_ms": 15,
                "timestamp": datetime.utcnow().isoformat(),
                "simulation": True
            }
    
    async def _execute_status_operation(self, device_id: int) -> Dict[str, Any]:
        """Execute status check operation"""
        if self.kernel_interface.is_available:
            return await self.kernel_interface.get_device_status(device_id)
        else:
            # Simulation mode
            device_info = get_device_info(device_id)
            return {
                "status": 0 if device_info["status"] == "ACTIVE" else 1,
                "device_info": device_info,
                "execution_time_ms": 5,
                "timestamp": datetime.utcnow().isoformat(),
                "simulation": True
            }
    
    async def _execute_reset_operation(self, device_id: int) -> Dict[str, Any]:
        """Execute reset operation"""
        if self.kernel_interface.is_available:
            return await self.kernel_interface.reset_device(device_id)
        else:
            # Simulation mode
            return {
                "status": 0,
                "operation": "RESET",
                "execution_time_ms": 100,
                "timestamp": datetime.utcnow().isoformat(),
                "simulation": True
            }
    
    async def assess_operation_risk(
        self, 
        device_id: int, 
        operation_type: str, 
        operation_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Assess risk level of operation"""
        return assess_operation_risk(device_id, operation_type, operation_data or {})
    
    async def get_all_device_states(self) -> List[Dict[str, Any]]:
        """Get current state of all devices"""
        states = []
        for device_id, state in self.device_states.items():
            device_info = get_device_info(device_id)
            states.append({
                **device_info,
                "is_active": state.is_active,
                "is_quarantined": state.is_quarantined,
                "last_response_time_ms": state.last_response_time_ms,
                "error_count": state.error_count,
                "success_count": state.success_count,
                "last_accessed": state.last_accessed.isoformat() if state.last_accessed else None
            })
        return states
    
    async def get_device_registry(self) -> List[Dict[str, Any]]:
        """Get complete device registry"""
        return await self.get_all_device_states()
    
    async def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics"""
        return self.performance_metrics.copy()
    
    def _update_performance_metrics(self, success: bool, execution_time_ms: int):
        """Update performance metrics"""
        self.performance_metrics["total_operations"] += 1
        
        if success:
            self.performance_metrics["successful_operations"] += 1
        else:
            self.performance_metrics["failed_operations"] += 1
        
        # Update error rate
        total_ops = self.performance_metrics["total_operations"]
        failed_ops = self.performance_metrics["failed_operations"]
        self.performance_metrics["error_rate"] = (failed_ops / total_ops) * 100 if total_ops > 0 else 0
        
        # Update average latency
        current_avg = self.performance_metrics["average_latency_ms"]
        self.performance_metrics["average_latency_ms"] = (
            (current_avg * (total_ops - 1) + execution_time_ms) / total_ops
        )
    
    async def health_check_all_devices(self):
        """Perform health check on all devices"""
        logger.info("Starting device health check...")
        
        health_results = {}
        
        for device_id in self.device_states:
            if not self.device_states[device_id].is_quarantined:
                try:
                    result = await self._execute_status_operation(device_id)
                    health_results[device_id] = {
                        "healthy": result.get("status", -1) == 0,
                        "response_time_ms": result.get("execution_time_ms", 0),
                        "last_check": datetime.utcnow()
                    }
                    
                    # Update device state
                    self.device_states[device_id].is_active = health_results[device_id]["healthy"]
                    self.device_states[device_id].last_response_time_ms = result.get("execution_time_ms", 0)
                    
                except Exception as e:
                    logger.warning(f"Health check failed for device {device_id:04X}: {e}")
                    health_results[device_id] = {
                        "healthy": False,
                        "error": str(e),
                        "last_check": datetime.utcnow()
                    }
                    self.device_states[device_id].is_active = False
        
        self.last_health_check = datetime.utcnow()
        logger.info(f"Health check complete: {len(health_results)} devices checked")
        
        return health_results
    
    async def health_check_loop(self):
        """Background health check loop"""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                await self.health_check_all_devices()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def cleanup(self):
        """Cleanup device controller"""
        logger.info("Cleaning up device controller...")
        self.kernel_interface.cleanup()
        self.active_operations.clear()
        logger.info("Device controller cleanup complete")


# Global device controller instance
device_controller = DeviceController()