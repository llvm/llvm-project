#!/usr/bin/env python3
"""
TPM2 Protocol Bridge for Compatibility Layer
Main integration component that coordinates PCR translation, ME wrapping, and military token validation

Author: C-INTERNAL Agent
Date: 2025-09-23
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
"""

import os
import struct
import logging
import time
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum

# Import our core components
from .pcr_translator import PCRAddressTranslator, PCRTranslationResult, PCRBankType
from .me_wrapper import MEInterfaceWrapper, MESessionContext
from .military_token_integration import DellMilitaryTokenManager, SecurityLevel, TokenValidationResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TPM2CommandCode(Enum):
    """TPM2 command codes for parsing"""
    TPM_CC_STARTUP = 0x00000144
    TPM_CC_SHUTDOWN = 0x00000145
    TPM_CC_GETRANDOM = 0x0000017B
    TPM_CC_PCR_READ = 0x0000017E
    TPM_CC_PCR_EXTEND = 0x00000182
    TPM_CC_CREATE_PRIMARY = 0x00000131
    TPM_CC_CREATE = 0x00000153
    TPM_CC_LOAD = 0x00000157
    TPM_CC_SIGN = 0x0000015D
    TPM_CC_QUOTE = 0x00000158

class BridgeOperationResult(Enum):
    """Result codes for bridge operations"""
    SUCCESS = 0
    ERROR_INVALID_COMMAND = 1
    ERROR_PCR_TRANSLATION = 2
    ERROR_TOKEN_VALIDATION = 3
    ERROR_ME_COMMUNICATION = 4
    ERROR_AUTHORIZATION = 5
    ERROR_TIMEOUT = 6

@dataclass
class TPM2Command:
    """Parsed TPM2 command structure"""
    tag: int
    size: int
    code: int
    handles: List[int]
    auth_area: bytes
    parameters: bytes
    raw_command: bytes

@dataclass
class BridgeResult:
    """Result of bridge operation"""
    success: bool
    result_code: BridgeOperationResult
    tpm_response: Optional[bytes]
    error_message: Optional[str]
    translation_info: Optional[Dict[str, Any]]
    execution_time: float

class TPM2ProtocolBridge:
    """
    Main TPM2 protocol bridge that coordinates all compatibility layer components
    Provides transparent TPM2 ↔ ME communication with military token authorization
    """

    # Command type mappings
    COMMAND_TYPE_MAP = {
        TPM2CommandCode.TPM_CC_STARTUP.value: "startup",
        TPM2CommandCode.TPM_CC_SHUTDOWN.value: "shutdown",
        TPM2CommandCode.TPM_CC_GETRANDOM.value: "getrandom",
        TPM2CommandCode.TPM_CC_PCR_READ.value: "pcrread",
        TPM2CommandCode.TPM_CC_PCR_EXTEND.value: "pcrextend",
        TPM2CommandCode.TPM_CC_CREATE_PRIMARY.value: "createkey",
        TPM2CommandCode.TPM_CC_CREATE.value: "createkey",
        TPM2CommandCode.TPM_CC_LOAD.value: "load",
        TPM2CommandCode.TPM_CC_SIGN.value: "sign",
        TPM2CommandCode.TPM_CC_QUOTE.value: "quote"
    }

    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize TPM2 protocol bridge

        Args:
            session_id: Optional session ID for audit logging
        """
        self.pcr_translator = PCRAddressTranslator()
        self.me_wrapper = MEInterfaceWrapper()
        self.token_manager = DellMilitaryTokenManager()

        # Set session ID for token manager audit logging
        if session_id:
            self.token_manager.set_session_id(session_id)

        self.current_me_session = None
        self.current_token_validation = None
        self.is_initialized = False
        self.command_count = 0

        logger.info("TPM2 Protocol Bridge initialized")

    def initialize(self, security_level: str = "UNCLASSIFIED") -> bool:
        """
        Initialize the protocol bridge with all components

        Args:
            security_level: Required security level for operations

        Returns:
            True if initialization successful
        """
        try:
            start_time = time.time()

            # 1. Validate military tokens
            logger.info("Validating military tokens...")
            self.current_token_validation = self.token_manager.validate_military_tokens()

            if not self.current_token_validation.success:
                logger.error("Military token validation failed")
                return False

            logger.info(f"Military tokens validated: {self.current_token_validation.authorization_level.name}")

            # 2. Initialize ME interface
            logger.info("Initializing ME interface...")
            if not self.me_wrapper.initialize_me_interface():
                logger.error("ME interface initialization failed")
                return False

            # 3. Establish ME session
            logger.info(f"Establishing ME session with {security_level} clearance...")
            self.current_me_session = self.me_wrapper.establish_me_session(security_level)

            if self.current_me_session is None:
                logger.error("ME session establishment failed")
                return False

            # 4. Create security handshake between tokens and ME
            logger.info("Creating security handshake...")
            handshake_data = self.token_manager.create_security_handshake(
                self.current_token_validation.tokens_validated
            )

            if not handshake_data:
                logger.error("Security handshake creation failed")
                return False

            self.is_initialized = True
            init_time = time.time() - start_time

            logger.info(f"Protocol bridge initialization complete ({init_time:.2f}s)")
            logger.info(f"Authorization level: {self.current_token_validation.authorization_level.name}")
            logger.info(f"Available operations: {len(self.current_token_validation.available_operations)}")

            return True

        except Exception as e:
            logger.error(f"Protocol bridge initialization error: {e}")
            return False

    def process_tpm_command(self, tpm_command: bytes) -> BridgeResult:
        """
        Process TPM2 command with full protocol translation

        Args:
            tpm_command: Raw TPM2 command bytes

        Returns:
            BridgeResult with translated response or error
        """
        start_time = time.time()
        self.command_count += 1

        try:
            if not self.is_initialized:
                return BridgeResult(
                    success=False,
                    result_code=BridgeOperationResult.ERROR_INVALID_COMMAND,
                    tmp_response=None,
                    error_message="Protocol bridge not initialized",
                    translation_info=None,
                    execution_time=time.time() - start_time
                )

            # 1. Parse TPM command
            parsed_command = self._parse_tpm_command(tpm_command)
            if parsed_command is None:
                return BridgeResult(
                    success=False,
                    result_code=BridgeOperationResult.ERROR_INVALID_COMMAND,
                    tpm_response=None,
                    error_message="Invalid TPM command format",
                    translation_info=None,
                    execution_time=time.time() - start_time
                )

            logger.debug(f"Processing TPM command: 0x{parsed_command.code:08X}")

            # 2. Determine operation type
            operation_type = self.COMMAND_TYPE_MAP.get(parsed_command.code, "unknown")

            # 3. Validate authorization for operation
            if not self.token_manager.validate_operation_authorization(operation_type):
                return BridgeResult(
                    success=False,
                    result_code=BridgeOperationResult.ERROR_AUTHORIZATION,
                    tpm_response=None,
                    error_message=f"Operation {operation_type} not authorized",
                    translation_info={'operation': operation_type},
                    execution_time=time.time() - start_time
                )

            # 4. Apply PCR translation if needed
            translated_command, translation_info = self._apply_pcr_translation(
                parsed_command, tpm_command
            )

            if translated_command is None:
                return BridgeResult(
                    success=False,
                    result_code=BridgeOperationResult.ERROR_PCR_TRANSLATION,
                    tpm_response=None,
                    error_message="PCR translation failed",
                    translation_info=translation_info,
                    execution_time=time.time() - start_time
                )

            # 5. Send command through ME interface
            me_response = self.me_wrapper.send_tpm_command_via_me(translated_command)

            if me_response is None:
                return BridgeResult(
                    success=False,
                    result_code=BridgeOperationResult.ERROR_ME_COMMUNICATION,
                    tpm_response=None,
                    error_message="ME communication failed",
                    translation_info=translation_info,
                    execution_time=time.time() - start_time
                )

            # 6. Apply reverse PCR translation to response if needed
            final_response = self._apply_reverse_pcr_translation(me_response, translation_info)

            execution_time = time.time() - start_time

            logger.info(f"TPM command processed successfully (cmd: {self.command_count}, time: {execution_time:.3f}s)")

            return BridgeResult(
                success=True,
                result_code=BridgeOperationResult.SUCCESS,
                tpm_response=final_response,
                error_message=None,
                translation_info=translation_info,
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error processing TPM command: {e}")

            return BridgeResult(
                success=False,
                result_code=BridgeOperationResult.ERROR_INVALID_COMMAND,
                tpm_response=None,
                error_message=f"Processing error: {str(e)}",
                translation_info=None,
                execution_time=execution_time
            )

    def get_bridge_status(self) -> Dict[str, Any]:
        """
        Get current bridge status and statistics

        Returns:
            Dictionary with bridge status information
        """
        return {
            'initialized': self.is_initialized,
            'command_count': self.command_count,
            'authorization_level': self.current_token_validation.authorization_level.name if self.current_token_validation else "NONE",
            'me_session_active': self.current_me_session is not None and self.current_me_session.is_active if self.current_me_session else False,
            'tokens_validated': len(self.current_token_validation.tokens_validated) if self.current_token_validation else 0,
            'available_operations': self.current_token_validation.available_operations if self.current_token_validation else [],
            'me_session_info': self.me_wrapper.get_session_info() if self.me_wrapper else {}
        }

    def cleanup(self):
        """Cleanup protocol bridge and all components"""
        try:
            logger.info("Cleaning up protocol bridge...")

            # Close ME session
            if self.current_me_session and self.me_wrapper:
                self.me_wrapper.close_me_session(self.current_me_session)

            # Cleanup ME wrapper
            if self.me_wrapper:
                self.me_wrapper.cleanup()

            # Clear caches
            if self.pcr_translator:
                self.pcr_translator.clear_cache()

            # Export audit log
            if self.token_manager:
                try:
                    audit_path = self.token_manager.export_audit_log()
                    logger.info(f"Audit log exported: {audit_path}")
                except Exception as e:
                    logger.warning(f"Failed to export audit log: {e}")

            self.is_initialized = False
            self.current_me_session = None
            self.current_token_validation = None

            logger.info("Protocol bridge cleanup complete")

        except Exception as e:
            logger.error(f"Error during bridge cleanup: {e}")

    # Private helper methods

    def _parse_tpm_command(self, tpm_command: bytes) -> Optional[TPM2Command]:
        """Parse TPM2 command structure"""
        try:
            if len(tpm_command) < 10:
                return None

            # Parse basic header
            tag, size, code = struct.unpack('>HII', tpm_command[:10])

            # Validate command size
            if size != len(tpm_command):
                logger.warning(f"TPM command size mismatch: header={size}, actual={len(tmp_command)}")

            # Extract handles and auth area (simplified parsing)
            handles = []
            auth_area = b''
            parameters = tpm_command[10:]

            return TPM2Command(
                tag=tag,
                size=size,
                code=code,
                handles=handles,
                auth_area=auth_area,
                parameters=parameters,
                raw_command=tpm_command
            )

        except Exception as e:
            logger.error(f"Error parsing TPM command: {e}")
            return None

    def _apply_pcr_translation(self, parsed_command: TPM2Command,
                             raw_command: bytes) -> Tuple[Optional[bytes], Dict[str, Any]]:
        """Apply PCR address translation to command if needed"""
        translation_info = {
            'pcr_translation_applied': False,
            'original_pcrs': [],
            'translated_pcrs': [],
            'operation': self.COMMAND_TYPE_MAP.get(parsed_command.code, "unknown")
        }

        try:
            # Check if command involves PCR operations
            if parsed_command.code not in [TPM2CommandCode.TPM_CC_PCR_READ.value,
                                         TPM2CommandCode.TPM_CC_PCR_EXTEND.value]:
                # No PCR translation needed
                return raw_command, translation_info

            # TODO: Implement actual PCR address translation within command
            # This would require detailed parsing of TPM command parameters
            # to identify PCR indices and translate them to hex format

            # For now, return original command
            translation_info['pcr_translation_applied'] = False
            return raw_command, translation_info

        except Exception as e:
            logger.error(f"Error applying PCR translation: {e}")
            return None, translation_info

    def _apply_reverse_pcr_translation(self, me_response: bytes,
                                     translation_info: Dict[str, Any]) -> bytes:
        """Apply reverse PCR translation to response if needed"""
        try:
            if not translation_info.get('pcr_translation_applied', False):
                return me_response

            # TODO: Implement reverse PCR translation for responses
            # This would translate hex PCR values back to decimal in response data

            return me_response

        except Exception as e:
            logger.error(f"Error applying reverse PCR translation: {e}")
            return me_response

    def refresh_authorization(self) -> bool:
        """Refresh military token validation and authorization"""
        try:
            logger.info("Refreshing authorization...")
            self.current_token_validation = self.token_manager.validate_military_tokens(force_refresh=True)

            if self.current_token_validation.success:
                logger.info(f"Authorization refreshed: {self.current_token_validation.authorization_level.name}")
                return True
            else:
                logger.error("Authorization refresh failed")
                return False

        except Exception as e:
            logger.error(f"Error refreshing authorization: {e}")
            return False


# Convenience functions for easy integration

def create_tpm2_bridge(session_id: Optional[str] = None,
                      security_level: str = "UNCLASSIFIED") -> Optional[TPM2ProtocolBridge]:
    """
    Create and initialize TPM2 protocol bridge

    Args:
        session_id: Optional session ID for audit logging
        security_level: Required security level

    Returns:
        Initialized TPM2ProtocolBridge or None if initialization failed
    """
    try:
        bridge = TPM2ProtocolBridge(session_id)

        if bridge.initialize(security_level):
            return bridge
        else:
            bridge.cleanup()
            return None

    except Exception as e:
        logger.error(f"Error creating TPM2 bridge: {e}")
        return None


def send_tpm_command_transparent(tpm_command: bytes,
                               session_id: Optional[str] = None,
                               security_level: str = "UNCLASSIFIED") -> Optional[bytes]:
    """
    Convenience function to send TPM command with transparent translation

    Args:
        tpm_command: Raw TPM2 command bytes
        session_id: Optional session ID
        security_level: Required security level

    Returns:
        TPM2 response bytes or None if error
    """
    try:
        bridge = create_tpm2_bridge(session_id, security_level)
        if bridge is None:
            return None

        result = bridge.process_tpm_command(tpm_command)
        bridge.cleanup()

        if result.success:
            return result.tpm_response
        else:
            logger.error(f"Command processing failed: {result.error_message}")
            return None

    except Exception as e:
        logger.error(f"Error in transparent command processing: {e}")
        return None


if __name__ == "__main__":
    # Test the protocol bridge
    print("=== TPM2 Protocol Bridge Test ===")

    try:
        # Create bridge
        bridge = TPM2ProtocolBridge("test_session_001")

        # Test initialization
        print("\n--- Bridge Initialization ---")
        if bridge.initialize("UNCLASSIFIED"):
            print("✓ Bridge initialization successful")

            # Show status
            status = bridge.get_bridge_status()
            print(f"Status: {status}")

            # Test command processing
            print("\n--- Command Processing ---")

            # Create test TPM2_Startup command
            test_startup_cmd = struct.pack('>HII', 0x8001, 12, TPM2CommandCode.TPM_CC_STARTUP.value) + b'\x00\x00'

            result = bridge.process_tpm_command(test_startup_cmd)

            if result.success:
                print(f"✓ Command processed successfully")
                print(f"Execution time: {result.execution_time:.3f}s")
                print(f"Response size: {len(result.tpm_response)} bytes" if result.tpm_response else "No response")
            else:
                print(f"✗ Command processing failed: {result.error_message}")

            # Test authorization refresh
            print("\n--- Authorization Refresh ---")
            if bridge.refresh_authorization():
                print("✓ Authorization refreshed successfully")
            else:
                print("✗ Authorization refresh failed")

        else:
            print("✗ Bridge initialization failed")

        # Cleanup
        print("\n--- Cleanup ---")
        bridge.cleanup()
        print("✓ Bridge cleanup completed")

    except Exception as e:
        print(f"✗ Test error: {e}")

    # Test convenience function
    print("\n--- Convenience Function Test ---")
    test_getrandom_cmd = struct.pack('>HII', 0x8001, 14, TPM2CommandCode.TPM_CC_GETRANDOM.value) + struct.pack('>H', 32)

    response = send_tpm_command_transparent(test_getrandom_cmd, "test_session_002", "UNCLASSIFIED")

    if response:
        print(f"✓ Transparent command successful: {len(response)} bytes")
    else:
        print("✗ Transparent command failed")