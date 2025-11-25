#!/usr/bin/env python3
"""
TPM Device Emulator for Compatibility Layer
Emulates /dev/tpm0 device interface for transparent tpm2-tools compatibility

Author: C-INTERNAL Agent
Date: 2025-09-23
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
"""

import os
import sys
import stat
import time
import logging
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from queue import Queue, Empty
import select
import errno

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.protocol_bridge import TPM2ProtocolBridge, create_tpm2_bridge
from core.constants import *

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EmulatedDevice:
    """Emulated TPM device information"""
    device_path: str
    device_major: int
    device_minor: int
    permissions: int
    is_active: bool
    created_time: float

class TPMDeviceEmulator:
    """
    TPM device emulator that creates /dev/tpm0 compatibility interface
    Intercepts TPM device operations and routes through compatibility layer
    """

    # TPM device configuration
    DEFAULT_DEVICE_PATH = "/dev/tpm0"
    TPM_DEVICE_MAJOR = 10  # Misc device major number
    TPM_DEVICE_MINOR = 224  # Standard TPM device minor
    DEVICE_PERMISSIONS = 0o666

    # Emulation settings
    MAX_CONCURRENT_SESSIONS = 10
    SESSION_TIMEOUT = 300  # 5 minutes
    COMMAND_QUEUE_SIZE = 100

    def __init__(self, device_path: str = None, security_level: str = "UNCLASSIFIED"):
        """
        Initialize TPM device emulator

        Args:
            device_path: Path for emulated device (default: /dev/tpm0)
            security_level: Security level for operations
        """
        self.device_path = device_path or self.DEFAULT_DEVICE_PATH
        self.security_level = security_level

        # Device state
        self.emulated_device = None
        self.is_running = False
        self.active_sessions = {}
        self.session_counter = 0

        # Threading and queues
        self.command_queue = Queue(maxsize=self.COMMAND_QUEUE_SIZE)
        self.response_queues = {}
        self.emulator_thread = None
        self.cleanup_thread = None

        # Protocol bridge
        self.protocol_bridge = None

        # Statistics
        self.stats = {
            'total_commands': 0,
            'successful_commands': 0,
            'failed_commands': 0,
            'sessions_created': 0,
            'sessions_closed': 0,
            'start_time': time.time()
        }

        logger.info(f"TPM Device Emulator initialized for {self.device_path}")

    def create_emulated_device(self) -> bool:
        """
        Create emulated TPM device node

        Returns:
            True if device created successfully
        """
        try:
            # Check if device already exists
            if os.path.exists(self.device_path):
                stat_info = os.stat(self.device_path)
                if stat.S_ISCHR(stat_info.st_mode):
                    logger.warning(f"TPM device already exists: {self.device_path}")
                    # Check if it's our emulated device or real device
                    if (stat.S_MAJOR(stat_info.st_rdev) == self.TPM_DEVICE_MAJOR and
                        stat.S_MINOR(stat_info.st_rdev) == self.TPM_DEVICE_MINOR):
                        logger.info("Using existing emulated TPM device")
                        return True
                    else:
                        logger.warning("Real TPM device exists - creating backup")
                        backup_path = f"{self.device_path}.backup"
                        if not os.path.exists(backup_path):
                            os.rename(self.device_path, backup_path)

            # Create device directory if needed
            device_dir = os.path.dirname(self.device_path)
            os.makedirs(device_dir, exist_ok=True)

            # Create character device node
            # Note: This requires root privileges in production
            try:
                os.mknod(self.device_path,
                        stat.S_IFCHR | self.DEVICE_PERMISSIONS,
                        os.makedev(self.TPM_DEVICE_MAJOR, self.TPM_DEVICE_MINOR))

                logger.info(f"Created emulated TPM device: {self.device_path}")

            except PermissionError:
                logger.warning("Insufficient permissions to create device node")
                logger.info("Creating simulation file instead")

                # Create simulation file for testing
                with open(f"{self.device_path}.sim", 'w') as f:
                    f.write("TPM_DEVICE_EMULATOR_SIMULATION\n")

                self.device_path = f"{self.device_path}.sim"

            # Store device information
            stat_info = os.stat(self.device_path)
            self.emulated_device = EmulatedDevice(
                device_path=self.device_path,
                device_major=self.TPM_DEVICE_MAJOR,
                device_minor=self.TPM_DEVICE_MINOR,
                permissions=self.DEVICE_PERMISSIONS,
                is_active=True,
                created_time=time.time()
            )

            return True

        except Exception as e:
            logger.error(f"Error creating emulated device: {e}")
            return False

    def start_emulation(self) -> bool:
        """
        Start TPM device emulation service

        Returns:
            True if emulation started successfully
        """
        try:
            if self.is_running:
                logger.warning("Emulation already running")
                return True

            # Create emulated device
            if not self.create_emulated_device():
                logger.error("Failed to create emulated device")
                return False

            # Initialize protocol bridge
            self.protocol_bridge = create_tpm2_bridge(
                session_id=f"emulator_{int(time.time())}",
                security_level=self.security_level
            )

            if self.protocol_bridge is None:
                logger.error("Failed to initialize protocol bridge")
                return False

            # Start emulator thread
            self.is_running = True
            self.emulator_thread = threading.Thread(target=self._emulator_worker, daemon=True)
            self.emulator_thread.start()

            # Start cleanup thread
            self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
            self.cleanup_thread.start()

            logger.info(f"TPM device emulation started: {self.device_path}")
            return True

        except Exception as e:
            logger.error(f"Error starting emulation: {e}")
            self.cleanup()
            return False

    def stop_emulation(self):
        """Stop TPM device emulation"""
        try:
            logger.info("Stopping TPM device emulation...")

            self.is_running = False

            # Wait for threads to finish
            if self.emulator_thread and self.emulator_thread.is_alive():
                self.emulator_thread.join(timeout=5.0)

            if self.cleanup_thread and self.cleanup_thread.is_alive():
                self.cleanup_thread.join(timeout=5.0)

            # Cleanup protocol bridge
            if self.protocol_bridge:
                self.protocol_bridge.cleanup()

            # Close all active sessions
            for session_id in list(self.active_sessions.keys()):
                self._close_session(session_id)

            logger.info("TPM device emulation stopped")

        except Exception as e:
            logger.error(f"Error stopping emulation: {e}")

    def cleanup(self):
        """Cleanup emulated device and resources"""
        try:
            # Stop emulation first
            self.stop_emulation()

            # Remove emulated device
            if self.emulated_device and os.path.exists(self.emulated_device.device_path):
                if self.emulated_device.device_path.endswith('.sim'):
                    os.remove(self.emulated_device.device_path)
                    logger.info("Removed simulation file")
                else:
                    # Restore backup if it exists
                    backup_path = f"{self.emulated_device.device_path}.backup"
                    if os.path.exists(backup_path):
                        os.remove(self.emulated_device.device_path)
                        os.rename(backup_path, self.emulated_device.device_path)
                        logger.info("Restored original TPM device")
                    else:
                        os.remove(self.emulated_device.device_path)
                        logger.info("Removed emulated device")

            self.emulated_device = None
            logger.info("Device emulator cleanup complete")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def process_tpm_command(self, command_data: bytes, session_id: str) -> bytes:
        """
        Process TPM command through emulation layer

        Args:
            command_data: Raw TPM command bytes
            session_id: Session identifier

        Returns:
            TPM response bytes
        """
        try:
            start_time = time.time()

            # Validate session
            if session_id not in self.active_sessions:
                logger.warning(f"Invalid session ID: {session_id}")
                return self._create_error_response(TPM_RC_FAILURE)

            # Update session activity
            self.active_sessions[session_id]['last_activity'] = time.time()

            # Process command through protocol bridge
            if self.protocol_bridge is None:
                logger.error("Protocol bridge not initialized")
                return self._create_error_response(TPM_RC_FAILURE)

            result = self.protocol_bridge.process_tpm_command(command_data)

            # Update statistics
            self.stats['total_commands'] += 1
            execution_time = time.time() - start_time

            if result.success:
                self.stats['successful_commands'] += 1
                logger.debug(f"Command processed successfully (session: {session_id}, time: {execution_time:.3f}s)")
                return result.tpm_response
            else:
                self.stats['failed_commands'] += 1
                logger.warning(f"Command processing failed: {result.error_message}")
                return self._create_error_response(TPM_RC_FAILURE)

        except Exception as e:
            logger.error(f"Error processing TPM command: {e}")
            self.stats['failed_commands'] += 1
            return self._create_error_response(TPM_RC_FAILURE)

    def create_session(self) -> str:
        """
        Create new emulated TPM session

        Returns:
            Session identifier
        """
        try:
            self.session_counter += 1
            session_id = f"emu_session_{self.session_counter:06d}"

            session_info = {
                'session_id': session_id,
                'created_time': time.time(),
                'last_activity': time.time(),
                'command_count': 0,
                'is_active': True
            }

            self.active_sessions[session_id] = session_info
            self.stats['sessions_created'] += 1

            logger.debug(f"Created emulation session: {session_id}")
            return session_id

        except Exception as e:
            logger.error(f"Error creating session: {e}")
            return None

    def get_emulation_status(self) -> Dict[str, Any]:
        """
        Get current emulation status and statistics

        Returns:
            Dictionary with emulation status
        """
        uptime = time.time() - self.stats['start_time']

        return {
            'is_running': self.is_running,
            'device_path': self.device_path,
            'device_active': self.emulated_device is not None and self.emulated_device.is_active,
            'security_level': self.security_level,
            'active_sessions': len(self.active_sessions),
            'statistics': {
                **self.stats,
                'uptime_seconds': uptime,
                'commands_per_second': self.stats['total_commands'] / uptime if uptime > 0 else 0,
                'success_rate': (self.stats['successful_commands'] / self.stats['total_commands']) * 100
                             if self.stats['total_commands'] > 0 else 0
            },
            'bridge_status': self.protocol_bridge.get_bridge_status() if self.protocol_bridge else None
        }

    # Private helper methods

    def _emulator_worker(self):
        """Main emulator worker thread"""
        logger.info("Emulator worker thread started")

        while self.is_running:
            try:
                # Process pending commands
                try:
                    # Get command from queue with timeout
                    command_item = self.command_queue.get(timeout=1.0)

                    command_data = command_item['data']
                    session_id = command_item['session_id']
                    response_queue = command_item['response_queue']

                    # Process command
                    response = self.process_tpm_command(command_data, session_id)

                    # Send response
                    response_queue.put(response)

                except Empty:
                    # No commands to process, continue loop
                    continue

            except Exception as e:
                logger.error(f"Error in emulator worker: {e}")
                time.sleep(0.1)

        logger.info("Emulator worker thread stopped")

    def _cleanup_worker(self):
        """Cleanup worker thread for session management"""
        logger.info("Cleanup worker thread started")

        while self.is_running:
            try:
                current_time = time.time()

                # Clean up expired sessions
                expired_sessions = []
                for session_id, session_info in self.active_sessions.items():
                    if current_time - session_info['last_activity'] > self.SESSION_TIMEOUT:
                        expired_sessions.append(session_id)

                for session_id in expired_sessions:
                    self._close_session(session_id)
                    logger.info(f"Cleaned up expired session: {session_id}")

                # Sleep before next cleanup cycle
                time.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Error in cleanup worker: {e}")
                time.sleep(30)

        logger.info("Cleanup worker thread stopped")

    def _close_session(self, session_id: str):
        """Close and cleanup session"""
        try:
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
                self.stats['sessions_closed'] += 1
                logger.debug(f"Closed session: {session_id}")

        except Exception as e:
            logger.error(f"Error closing session {session_id}: {e}")

    def _create_error_response(self, error_code: int) -> bytes:
        """Create TPM error response"""
        try:
            # Create minimal TPM error response
            response = struct.pack('>HII', TPM_ST_NO_SESSIONS, 10, error_code)
            return response

        except Exception as e:
            logger.error(f"Error creating error response: {e}")
            return b'\x80\x01\x00\x0a\x00\x00\x01\x01'  # Basic failure response


class TPMDeviceInterceptor:
    """
    Intercepts system calls to TPM device and routes through emulator
    Note: This is a conceptual implementation - real interception would require
    kernel module or LD_PRELOAD library in C
    """

    def __init__(self, emulator: TPMDeviceEmulator):
        """
        Initialize device interceptor

        Args:
            emulator: TPM device emulator instance
        """
        self.emulator = emulator
        self.intercepted_operations = {}

    def intercept_device_access(self, operation: str, *args, **kwargs) -> Any:
        """
        Intercept device access operations

        Args:
            operation: Operation type (open, read, write, close, ioctl)
            *args, **kwargs: Operation arguments

        Returns:
            Operation result
        """
        try:
            logger.debug(f"Intercepted {operation} operation")

            if operation == "open":
                return self._handle_open(*args, **kwargs)
            elif operation == "read":
                return self._handle_read(*args, **kwargs)
            elif operation == "write":
                return self._handle_write(*args, **kwargs)
            elif operation == "close":
                return self._handle_close(*args, **kwargs)
            elif operation == "ioctl":
                return self._handle_ioctl(*args, **kwargs)
            else:
                logger.warning(f"Unknown operation: {operation}")
                return -1

        except Exception as e:
            logger.error(f"Error intercepting {operation}: {e}")
            return -1

    def _handle_open(self, path: str, flags: int) -> int:
        """Handle device open operation"""
        if path == self.emulator.device_path:
            session_id = self.emulator.create_session()
            if session_id:
                # Return session ID as file descriptor
                fd = hash(session_id) & 0x7FFFFFFF
                self.intercepted_operations[fd] = {
                    'session_id': session_id,
                    'operation': 'open',
                    'path': path
                }
                return fd
            else:
                return -1
        else:
            # Pass through to real system call
            return os.open(path, flags)

    def _handle_write(self, fd: int, data: bytes) -> int:
        """Handle device write operation"""
        if fd in self.intercepted_operations:
            session_id = self.intercepted_operations[fd]['session_id']

            # Queue command for processing
            response_queue = Queue()
            command_item = {
                'data': data,
                'session_id': session_id,
                'response_queue': response_queue
            }

            try:
                self.emulator.command_queue.put(command_item, timeout=5.0)
                # Store response queue for later read
                self.intercepted_operations[fd]['response_queue'] = response_queue
                return len(data)  # Simulate successful write

            except Exception as e:
                logger.error(f"Error queuing command: {e}")
                return -1
        else:
            # Pass through to real system call
            return os.write(fd, data)

    def _handle_read(self, fd: int, size: int) -> bytes:
        """Handle device read operation"""
        if fd in self.intercepted_operations:
            response_queue = self.intercepted_operations[fd].get('response_queue')

            if response_queue:
                try:
                    # Wait for response
                    response = response_queue.get(timeout=10.0)
                    return response
                except Empty:
                    logger.error("Response timeout")
                    return b''
            else:
                logger.error("No response queue available")
                return b''
        else:
            # Pass through to real system call
            return os.read(fd, size)

    def _handle_close(self, fd: int) -> int:
        """Handle device close operation"""
        if fd in self.intercepted_operations:
            session_id = self.intercepted_operations[fd]['session_id']
            self.emulator._close_session(session_id)
            del self.intercepted_operations[fd]
            return 0
        else:
            # Pass through to real system call
            return os.close(fd)

    def _handle_ioctl(self, fd: int, request: int, arg: Any) -> int:
        """Handle device ioctl operation"""
        if fd in self.intercepted_operations:
            # Handle TPM-specific ioctl operations
            logger.debug(f"TPM ioctl: request=0x{request:08X}")
            # TODO: Implement TPM ioctl handling
            return 0
        else:
            # Pass through to real system call
            import fcntl
            return fcntl.ioctl(fd, request, arg)


# Convenience functions
def start_tpm_device_emulation(device_path: str = None,
                              security_level: str = "UNCLASSIFIED") -> Optional[TPMDeviceEmulator]:
    """
    Start TPM device emulation

    Args:
        device_path: Path for emulated device
        security_level: Security level for operations

    Returns:
        TPMDeviceEmulator instance or None if failed
    """
    try:
        emulator = TPMDeviceEmulator(device_path, security_level)

        if emulator.start_emulation():
            return emulator
        else:
            emulator.cleanup()
            return None

    except Exception as e:
        logger.error(f"Error starting TPM device emulation: {e}")
        return None


if __name__ == "__main__":
    # Test the device emulator
    print("=== TPM Device Emulator Test ===")

    try:
        # Start emulation
        print("\n--- Starting Emulation ---")
        emulator = start_tpm_device_emulation("/dev/tpm0.test", "UNCLASSIFIED")

        if emulator:
            print("✓ Device emulation started successfully")

            # Show status
            status = emulator.get_emulation_status()
            print(f"Status: {status}")

            # Simulate some operations
            print("\n--- Simulating Operations ---")

            session_id = emulator.create_session()
            if session_id:
                print(f"✓ Session created: {session_id}")

                # Test command processing
                test_command = struct.pack('>HII', TPM_ST_NO_SESSIONS, 12, TPM_CC_STARTUP) + b'\x00\x00'
                response = emulator.process_tpm_command(test_command, session_id)

                if response:
                    print(f"✓ Command processed: {len(response)} bytes response")
                else:
                    print("✗ Command processing failed")

            else:
                print("✗ Session creation failed")

            # Show final status
            print("\n--- Final Status ---")
            final_status = emulator.get_emulation_status()
            print(f"Commands processed: {final_status['statistics']['total_commands']}")
            print(f"Success rate: {final_status['statistics']['success_rate']:.1f}%")

        else:
            print("✗ Device emulation failed to start")

        # Cleanup
        print("\n--- Cleanup ---")
        if emulator:
            emulator.cleanup()
            print("✓ Emulator cleanup completed")

    except Exception as e:
        print(f"✗ Test error: {e}")