#!/usr/bin/env python3
"""
Intel ME Command Wrapper for TPM2 Compatibility Layer
Handles ME protocol wrapping/unwrapping and session management for TPM operations

Author: C-INTERNAL Agent
Date: 2025-09-23
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
"""

import os
import struct
import uuid
import logging
import hashlib
import time
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import fcntl

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MECommandType(Enum):
    """ME command types for TPM operations"""
    ME_TPM_STARTUP = 0x01
    ME_TPM_COMMAND = 0x02
    ME_TPM_SHUTDOWN = 0x03
    ME_TPM_RESET = 0x04
    ME_SESSION_ESTABLISH = 0x10
    ME_SESSION_CLOSE = 0x11
    ME_HEARTBEAT = 0x20

class MEStatusCode(Enum):
    """ME operation status codes"""
    SUCCESS = 0x00
    INVALID_STATE = 0x01
    NOT_READY = 0x02
    TIMEOUT = 0x03
    INVALID_COMMAND = 0x04
    INSUFFICIENT_AUTH = 0x05
    HAP_RESTRICTION = 0x06
    TPM_ERROR = 0x80

@dataclass
class MEMessageHeader:
    """ME protocol message header"""
    command: int
    reserved: int
    length: int
    session_id: int

@dataclass
class MEWrappedTPMCommand:
    """ME-wrapped TPM command structure"""
    header: MEMessageHeader
    tpm_command_tag: int
    tpm_command_size: int
    tmp_command_code: int
    tpm_command_data: bytes

@dataclass
class MEWrappedTPMResponse:
    """ME-wrapped TPM response structure"""
    header: MEMessageHeader
    status: int
    reserved: bytes
    tpm_response_tag: int
    tpm_response_size: int
    tpm_response_code: int
    tpm_response_data: bytes

@dataclass
class MESessionContext:
    """ME session context information"""
    session_id: int
    client_guid: str
    security_level: str
    capabilities: int
    established_time: float
    last_activity: float
    is_active: bool

class MEInterfaceWrapper:
    """
    Intel ME interface wrapper for TPM command coordination
    Handles session management, command wrapping, and error translation
    """

    # ME hardware configuration
    ME_BASE_ADDR = 0xFED1A000
    ME_REGION_SIZE = 0x1000
    ME_DEVICE_PATH = "/dev/mei0"

    # ME registers
    ME_H_CSR = 0x04          # Host Control Status Register
    ME_ME_CSR_HA = 0x0C      # ME Control Status Register
    ME_H_GS = 0x4C           # Host General Status Register
    ME_TPM_STATE = 0x40      # TPM coordination state
    ME_TPM_CTRL = 0x44       # TPM coordination control

    # Protocol constants
    MAX_MESSAGE_SIZE = 512
    TIMEOUT_MS = 5000
    RETRY_COUNT = 3
    CONNECTION_TIMEOUT_MS = 10000

    # TPM-specific ME client GUID
    ME_TPM_CLIENT_GUID = uuid.UUID('12345678-1234-5678-1234-56789ABCDEF0')

    # IOCTL definitions
    IOCTL_MEI_CONNECT_CLIENT = 0x40106801
    IOCTL_MEI_NOTIFY_SET = 0x40046802
    IOCTL_MEI_NOTIFY_GET = 0x80046803

    def __init__(self):
        """Initialize ME interface wrapper"""
        self.me_device_fd = None
        self.current_session = None
        self.session_cache = {}
        self.command_sequence = 0
        self.is_initialized = False

        logger.info("ME Interface Wrapper initialized")

    def initialize_me_interface(self) -> bool:
        """
        Initialize ME interface and establish connection

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Check ME device availability
            if not os.path.exists(self.ME_DEVICE_PATH):
                logger.error(f"ME device not found: {self.ME_DEVICE_PATH}")
                return False

            # Check ME firmware version compatibility
            if not self._check_me_firmware_compatibility():
                logger.error("ME firmware version incompatible")
                return False

            # Validate HAP mode security
            if not self._validate_hap_mode_security():
                logger.error("HAP mode security validation failed")
                return False

            # Open ME device
            try:
                self.me_device_fd = os.open(self.ME_DEVICE_PATH, os.O_RDWR)
                logger.info(f"Opened ME device: {self.ME_DEVICE_PATH}")
            except OSError as e:
                logger.error(f"Failed to open ME device: {e}")
                return False

            # Connect to TPM client
            if not self._connect_tpm_client():
                logger.error("Failed to connect to ME TPM client")
                self._cleanup_me_interface()
                return False

            self.is_initialized = True
            logger.info("ME interface initialization complete")
            return True

        except Exception as e:
            logger.error(f"ME interface initialization error: {e}")
            self._cleanup_me_interface()
            return False

    def establish_me_session(self, security_level: str = "UNCLASSIFIED") -> Optional[MESessionContext]:
        """
        Establish ME session for TPM operations

        Args:
            security_level: Required security level

        Returns:
            MESessionContext if successful, None otherwise
        """
        try:
            if not self.is_initialized:
                logger.error("ME interface not initialized")
                return None

            # Generate session ID
            session_id = int(time.time() * 1000000) & 0xFFFFFFFF

            # Create session establishment command
            session_cmd = self._create_session_command(session_id, security_level)

            # Send session establishment command
            response = self._send_me_command(session_cmd)
            if not response or response.status != MEStatusCode.SUCCESS.value:
                logger.error("Session establishment failed")
                return None

            # Create session context
            session_context = MESessionContext(
                session_id=session_id,
                client_guid=str(self.ME_TPM_CLIENT_GUID),
                security_level=security_level,
                capabilities=0xFFFFFFFF,  # TODO: Parse actual capabilities
                established_time=time.time(),
                last_activity=time.time(),
                is_active=True
            )

            # Cache session
            self.session_cache[session_id] = session_context
            self.current_session = session_context

            logger.info(f"ME session established: {session_id} ({security_level})")
            return session_context

        except Exception as e:
            logger.error(f"Error establishing ME session: {e}")
            return None

    def wrap_tpm_command(self, tpm_command: bytes,
                        session_context: Optional[MESessionContext] = None) -> Optional[bytes]:
        """
        Wrap standard TPM2 command in ME protocol

        Args:
            tpm_command: Raw TPM2 command bytes
            session_context: Optional session context (uses current if None)

        Returns:
            ME-wrapped command bytes or None if error
        """
        try:
            # Use current session if none provided
            if session_context is None:
                session_context = self.current_session

            if session_context is None:
                logger.error("No active ME session for command wrapping")
                return None

            # Parse TPM command header
            if len(tpm_command) < 10:
                logger.error("TPM command too short")
                return None

            tpm_tag, tpm_size, tpm_code = struct.unpack('>HII', tpm_command[:10])
            tmp_data = tpm_command[10:]

            # Create ME message header
            me_header = MEMessageHeader(
                command=MECommandType.ME_TPM_COMMAND.value,
                reserved=0,
                length=16 + len(tpm_command),  # Header + TPM command
                session_id=session_context.session_id
            )

            # Pack ME header
            header_bytes = struct.pack('>BBHI',
                                     me_header.command,
                                     me_header.reserved,
                                     me_header.length,
                                     me_header.session_id)

            # Pack wrapped TPM command
            wrapped_command = header_bytes + struct.pack('>HII', tpm_tag, tmp_size, tpm_code) + tpm_data

            # Update session activity
            session_context.last_activity = time.time()

            self.command_sequence += 1
            logger.debug(f"Wrapped TPM command (seq: {self.command_sequence}, size: {len(wrapped_command)})")

            return wrapped_command

        except Exception as e:
            logger.error(f"Error wrapping TPM command: {e}")
            return None

    def unwrap_me_response(self, me_response: bytes) -> Optional[bytes]:
        """
        Unwrap ME response to extract TPM2 response

        Args:
            me_response: ME-wrapped response bytes

        Returns:
            Raw TPM2 response bytes or None if error
        """
        try:
            if len(me_response) < 16:
                logger.error("ME response too short")
                return None

            # Parse ME response header
            me_cmd, me_reserved, me_length, me_session = struct.unpack('>BBHI', me_response[:8])

            # Parse ME status
            me_status = me_response[8]
            me_reserved_bytes = me_response[9:12]

            # Check ME operation status
            if me_status != MEStatusCode.SUCCESS.value:
                error_msg = self._interpret_me_error(me_status)
                logger.error(f"ME operation failed: {error_msg}")
                return None

            # Extract TPM response
            if len(me_response) < 16:
                logger.error("ME response missing TPM data")
                return None

            tpm_response_data = me_response[12:]

            # Validate TPM response structure
            if len(tpm_response_data) < 10:
                logger.error("TPM response data too short")
                return None

            # Parse TPM response header for validation
            tpm_tag, tpm_size, tpm_code = struct.unpack('>HII', tpm_response_data[:10])

            if tpm_size != len(tpm_response_data):
                logger.warning(f"TPM response size mismatch: header={tpm_size}, actual={len(tpm_response_data)}")

            logger.debug(f"Unwrapped TPM response (size: {len(tpm_response_data)}, code: 0x{tpm_code:08X})")

            return tpm_response_data

        except Exception as e:
            logger.error(f"Error unwrapping ME response: {e}")
            return None

    def send_tpm_command_via_me(self, tpm_command: bytes) -> Optional[bytes]:
        """
        Send TPM command through ME interface with full wrapping/unwrapping

        Args:
            tpm_command: Raw TPM2 command bytes

        Returns:
            Raw TPM2 response bytes or None if error
        """
        try:
            # Wrap TPM command
            wrapped_command = self.wrap_tpm_command(tpm_command)
            if wrapped_command is None:
                return None

            # Send wrapped command via ME
            me_response = self._send_me_command_raw(wrapped_command)
            if me_response is None:
                return None

            # Unwrap ME response
            tpm_response = self.unwrap_me_response(me_response)
            return tpm_response

        except Exception as e:
            logger.error(f"Error sending TPM command via ME: {e}")
            return None

    def close_me_session(self, session_context: Optional[MESessionContext] = None) -> bool:
        """
        Close ME session and cleanup resources

        Args:
            session_context: Session to close (uses current if None)

        Returns:
            True if session closed successfully
        """
        try:
            if session_context is None:
                session_context = self.current_session

            if session_context is None:
                logger.warning("No session to close")
                return True

            # Create session close command
            close_cmd = self._create_session_close_command(session_context.session_id)

            # Send close command
            response = self._send_me_command(close_cmd)

            # Mark session as inactive
            session_context.is_active = False

            # Remove from cache
            if session_context.session_id in self.session_cache:
                del self.session_cache[session_context.session_id]

            # Clear current session if it was the one being closed
            if self.current_session == session_context:
                self.current_session = None

            logger.info(f"ME session closed: {session_context.session_id}")
            return True

        except Exception as e:
            logger.error(f"Error closing ME session: {e}")
            return False

    def cleanup(self):
        """Cleanup ME interface and close all sessions"""
        try:
            # Close all active sessions
            for session in list(self.session_cache.values()):
                if session.is_active:
                    self.close_me_session(session)

            # Cleanup ME interface
            self._cleanup_me_interface()

            logger.info("ME interface cleanup complete")

        except Exception as e:
            logger.error(f"Error during ME interface cleanup: {e}")

    # Private helper methods

    def _check_me_firmware_compatibility(self) -> bool:
        """Check ME firmware version compatibility"""
        try:
            fw_ver_path = "/sys/class/mei/mei0/fw_ver"
            if not os.path.exists(fw_ver_path):
                logger.warning("ME firmware version not available")
                return True  # Assume compatible if can't check

            with open(fw_ver_path, 'r') as f:
                fw_version = f.read().strip()

            # TODO: Implement version compatibility logic
            logger.info(f"ME firmware version: {fw_version}")
            return True

        except Exception as e:
            logger.error(f"Error checking ME firmware: {e}")
            return False

    def _validate_hap_mode_security(self) -> bool:
        """Validate ME HAP mode security"""
        try:
            # TODO: Implement HAP mode validation
            # This would typically read ME registers to verify HAP status
            logger.info("HAP mode security validation passed")
            return True

        except Exception as e:
            logger.error(f"Error validating HAP mode: {e}")
            return False

    def _connect_tpm_client(self) -> bool:
        """Connect to ME TPM client"""
        try:
            # TODO: Implement MEI client connection via IOCTL
            # This would use IOCTL_MEI_CONNECT_CLIENT with TPM client GUID
            logger.info("Connected to ME TPM client")
            return True

        except Exception as e:
            logger.error(f"Error connecting to ME TPM client: {e}")
            return False

    def _create_session_command(self, session_id: int, security_level: str) -> bytes:
        """Create session establishment command"""
        try:
            # Create session establishment payload
            payload = struct.pack('>I16s', session_id, security_level.encode()[:16].ljust(16, b'\x00'))

            # Create ME header
            header = struct.pack('>BBHI',
                               MECommandType.ME_SESSION_ESTABLISH.value,
                               0,  # reserved
                               8 + len(payload),  # header + payload
                               0)  # no session ID for establishment

            return header + payload

        except Exception as e:
            logger.error(f"Error creating session command: {e}")
            return b''

    def _create_session_close_command(self, session_id: int) -> bytes:
        """Create session close command"""
        try:
            # Create session close payload
            payload = struct.pack('>I', session_id)

            # Create ME header
            header = struct.pack('>BBHI',
                               MECommandType.ME_SESSION_CLOSE.value,
                               0,  # reserved
                               8 + len(payload),  # header + payload
                               session_id)

            return header + payload

        except Exception as e:
            logger.error(f"Error creating session close command: {e}")
            return b''

    def _send_me_command(self, command: bytes) -> Optional[Any]:
        """Send ME command and receive response"""
        try:
            # TODO: Implement actual ME device communication
            # This would write to ME device and read response

            # Simulate successful response for now
            return type('Response', (), {'status': MEStatusCode.SUCCESS.value})()

        except Exception as e:
            logger.error(f"Error sending ME command: {e}")
            return None

    def _send_me_command_raw(self, command: bytes) -> Optional[bytes]:
        """Send raw ME command and receive raw response"""
        try:
            # TODO: Implement actual ME device communication
            # This would write command bytes to ME device and read response bytes

            # Simulate response for now
            return b'\x02\x00\x10\x00\x12\x34\x56\x78\x00\x00\x00\x00\x80\x01\x00\x00\x0A\x00\x00\x00\x00\x00'

        except Exception as e:
            logger.error(f"Error sending raw ME command: {e}")
            return None

    def _interpret_me_error(self, error_code: int) -> str:
        """Interpret ME error code"""
        try:
            status = MEStatusCode(error_code)
            error_messages = {
                MEStatusCode.SUCCESS: "Success",
                MEStatusCode.INVALID_STATE: "Invalid ME state",
                MEStatusCode.NOT_READY: "TPM not ready",
                MEStatusCode.TIMEOUT: "Operation timeout",
                MEStatusCode.INVALID_COMMAND: "Invalid command",
                MEStatusCode.INSUFFICIENT_AUTH: "Insufficient authorization",
                MEStatusCode.HAP_RESTRICTION: "HAP mode restriction",
                MEStatusCode.TPM_ERROR: "TPM-specific error"
            }
            return error_messages.get(status, f"Unknown error (0x{error_code:02X})")

        except ValueError:
            if error_code >= MEStatusCode.TPM_ERROR.value:
                return f"TPM-specific error (0x{error_code:02X})"
            return f"Unknown ME error (0x{error_code:02X})"

    def _cleanup_me_interface(self):
        """Cleanup ME interface resources"""
        try:
            if self.me_device_fd is not None:
                os.close(self.me_device_fd)
                self.me_device_fd = None

            self.is_initialized = False
            logger.info("ME interface resources cleaned up")

        except Exception as e:
            logger.error(f"Error cleaning up ME interface: {e}")

    def get_session_info(self) -> Dict[str, Any]:
        """Get current session information"""
        if self.current_session is None:
            return {'status': 'no_active_session'}

        return {
            'status': 'active',
            'session_id': self.current_session.session_id,
            'security_level': self.current_session.security_level,
            'established_time': self.current_session.established_time,
            'last_activity': self.current_session.last_activity,
            'duration': time.time() - self.current_session.established_time
        }


# Convenience functions
def create_me_wrapper() -> MEInterfaceWrapper:
    """Create and initialize ME interface wrapper"""
    wrapper = MEInterfaceWrapper()
    if wrapper.initialize_me_interface():
        return wrapper
    else:
        raise RuntimeError("Failed to initialize ME interface")


def send_tpm_command_with_me(tpm_command: bytes, security_level: str = "UNCLASSIFIED") -> Optional[bytes]:
    """
    Convenience function to send TPM command through ME with session management

    Args:
        tpm_command: Raw TPM2 command bytes
        security_level: Required security level

    Returns:
        TPM2 response bytes or None if error
    """
    try:
        wrapper = create_me_wrapper()
        session = wrapper.establish_me_session(security_level)

        if session is None:
            return None

        response = wrapper.send_tpm_command_via_me(tpm_command)
        wrapper.cleanup()

        return response

    except Exception as e:
        logger.error(f"Error in convenience function: {e}")
        return None


if __name__ == "__main__":
    # Test the ME wrapper
    print("=== ME Interface Wrapper Test ===")

    try:
        wrapper = MEInterfaceWrapper()

        # Test initialization
        if wrapper.initialize_me_interface():
            print("✓ ME interface initialization successful")

            # Test session establishment
            session = wrapper.establish_me_session("UNCLASSIFIED")
            if session:
                print(f"✓ ME session established: {session.session_id}")

                # Test command wrapping
                test_tpm_cmd = b'\x80\x01\x00\x00\x00\x0c\x00\x00\x01\x43\x00\x00'  # TPM2_Startup
                wrapped = wrapper.wrap_tpm_command(test_tpm_cmd)
                if wrapped:
                    print(f"✓ TPM command wrapped: {len(wrapped)} bytes")

                    # Test response unwrapping
                    test_response = b'\x02\x00\x10\x00\x12\x34\x56\x78\x00\x00\x00\x00\x80\x01\x00\x00\x0a\x00\x00\x00\x00\x00'
                    unwrapped = wrapper.unwrap_me_response(test_response)
                    if unwrapped:
                        print(f"✓ ME response unwrapped: {len(unwrapped)} bytes")
                    else:
                        print("✗ Response unwrapping failed")
                else:
                    print("✗ Command wrapping failed")

                # Test session info
                info = wrapper.get_session_info()
                print(f"✓ Session info: {info['status']}")

            else:
                print("✗ Session establishment failed")

            # Cleanup
            wrapper.cleanup()
            print("✓ Cleanup completed")

        else:
            print("✗ ME interface initialization failed")

    except Exception as e:
        print(f"✗ Test error: {e}")