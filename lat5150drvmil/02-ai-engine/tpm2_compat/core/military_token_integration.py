#!/usr/bin/env python3
"""
Military Token Integration for TPM2 Compatibility Layer
Implements Dell military token validation and authorization for TPM operations

Author: C-INTERNAL Agent
Date: 2025-09-23
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
"""

import os
import json
import time
import struct
import hashlib
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timezone

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Military security classification levels"""
    NONE = 0
    UNCLASSIFIED = 1
    CONFIDENTIAL = 2
    SECRET = 3
    TOP_SECRET = 4

class TokenState(Enum):
    """Military token states"""
    INACTIVE = 0x00000000
    INITIALIZING = 0x00000001
    READY = 0x00000002
    ACTIVE = 0x00000003
    ERROR = 0x000000FF
    MILITARY_PATTERN = 0x44000000
    DEBUG_PATTERN = 0xDEADBEEF

@dataclass
class MilitaryTokenInfo:
    """Information about a military token"""
    token_id: str
    name: str
    security_level: SecurityLevel
    description: str
    required_for: List[str]
    value: Optional[int] = None
    status: Optional[str] = None
    last_read: Optional[float] = None

@dataclass
class TokenValidationResult:
    """Result of token validation"""
    success: bool
    tokens_validated: List[Dict[str, Any]]
    tokens_missing: List[str]
    authorization_level: SecurityLevel
    available_operations: List[str]
    security_warnings: List[str]
    validation_timestamp: float

@dataclass
class SecurityAuditEvent:
    """Security audit event for military compliance"""
    timestamp: str
    session_id: Optional[str]
    event_type: str
    operation: str
    details: str
    platform: str
    authorization_level: str
    tokens_used: List[str]

class DellMilitaryTokenManager:
    """
    Dell military token manager for TPM2 compatibility layer
    Handles token validation, authorization, and security auditing
    """

    # Military token registry
    MILITARY_TOKENS = {
        "049e": MilitaryTokenInfo(
            token_id="049e",
            name="MIL-SPEC Primary Authorization",
            security_level=SecurityLevel.UNCLASSIFIED,
            description="Base military system authorization",
            required_for=["basic_tpm_operations", "device_enumeration"]
        ),
        "049f": MilitaryTokenInfo(
            token_id="049f",
            name="MIL-SPEC Secondary Validation",
            security_level=SecurityLevel.CONFIDENTIAL,
            description="Enhanced security validation",
            required_for=["crypto_operations", "key_management"]
        ),
        "04a0": MilitaryTokenInfo(
            token_id="04a0",
            name="Hardware Feature Activation",
            security_level=SecurityLevel.CONFIDENTIAL,
            description="Hardware security feature control",
            required_for=["advanced_crypto", "hardware_attestation"]
        ),
        "04a1": MilitaryTokenInfo(
            token_id="04a1",
            name="Advanced Security Features",
            security_level=SecurityLevel.SECRET,
            description="Military-grade crypto operations",
            required_for=["classified_operations", "nsa_algorithms"]
        ),
        "04a2": MilitaryTokenInfo(
            token_id="04a2",
            name="System Integration Control",
            security_level=SecurityLevel.SECRET,
            description="Full system integration control",
            required_for=["me_coordination", "platform_attestation"]
        ),
        "04a3": MilitaryTokenInfo(
            token_id="04a3",
            name="Military Validation Token",
            security_level=SecurityLevel.TOP_SECRET,
            description="Maximum security authorization",
            required_for=["quantum_crypto", "top_secret_operations"]
        )
    }

    # Security level requirements for operations
    SECURITY_LEVELS = {
        SecurityLevel.UNCLASSIFIED: {
            'required_tokens': ['049e'],
            'encryption': 'AES-128',
            'key_strength': 2048,
            'audit_level': 'basic',
            'tpm_operations': ['startup', 'getrandom', 'pcrread']
        },
        SecurityLevel.CONFIDENTIAL: {
            'required_tokens': ['049e', '049f'],
            'encryption': 'AES-256',
            'key_strength': 3072,
            'audit_level': 'enhanced',
            'tpm_operations': ['startup', 'getrandom', 'pcrread', 'pcrextend', 'createkey']
        },
        SecurityLevel.SECRET: {
            'required_tokens': ['049e', '049f', '04a0', '04a1'],
            'encryption': 'AES-256-GCM',
            'key_strength': 4096,
            'audit_level': 'comprehensive',
            'tpm_operations': ['all_standard', 'advanced_crypto', 'attestation']
        },
        SecurityLevel.TOP_SECRET: {
            'required_tokens': ['049e', '049f', '04a0', '04a1', '04a2', '04a3'],
            'encryption': 'Post-quantum',
            'key_strength': 'quantum_resistant',
            'audit_level': 'maximum',
            'tpm_operations': ['all_operations', 'quantum_crypto', 'nsa_algorithms']
        }
    }

    # TPM operation authorization matrix
    TPM_AUTHORIZATION_MATRIX = {
        'startup': {
            'required_tokens': ['049e'],
            'security_level': SecurityLevel.UNCLASSIFIED,
            'me_coordination': True,
            'description': 'Initialize TPM through ME coordination'
        },
        'getrandom': {
            'required_tokens': ['049e'],
            'security_level': SecurityLevel.UNCLASSIFIED,
            'me_coordination': False,
            'description': 'Generate random numbers'
        },
        'pcrread': {
            'required_tokens': ['049e'],
            'security_level': SecurityLevel.UNCLASSIFIED,
            'me_coordination': True,
            'hex_pcr_support': True,
            'description': 'Read PCR values including hex ranges'
        },
        'pcrextend': {
            'required_tokens': ['049e', '049f'],
            'security_level': SecurityLevel.CONFIDENTIAL,
            'me_coordination': True,
            'hex_pcr_support': True,
            'description': 'Extend PCR values'
        },
        'createkey': {
            'required_tokens': ['049e', '049f', '04a0'],
            'security_level': SecurityLevel.CONFIDENTIAL,
            'me_coordination': True,
            'key_types': ['rsa2048', 'rsa3072', 'ecc256', 'ecc384'],
            'description': 'Create cryptographic keys'
        },
        'sign': {
            'required_tokens': ['049e', '049f', '04a0', '04a1'],
            'security_level': SecurityLevel.SECRET,
            'me_coordination': True,
            'algorithms': ['rsa-pss', 'ecdsa', 'ecdaa'],
            'description': 'Digital signing operations'
        },
        'quote': {
            'required_tokens': ['049e', '049f', '04a0', '04a1', '04a2'],
            'security_level': SecurityLevel.SECRET,
            'me_coordination': True,
            'attestation_types': ['platform', 'application', 'full_system'],
            'description': 'Platform attestation operations'
        },
        'nsa_algorithms': {
            'required_tokens': ['049e', '049f', '04a0', '04a1', '04a2', '04a3'],
            'security_level': SecurityLevel.TOP_SECRET,
            'me_coordination': True,
            'algorithms': ['suite_b', 'sha3', 'post_quantum'],
            'description': 'NSA Suite B and advanced algorithms'
        }
    }

    def __init__(self, audit_log_path: str = "/var/log/military_tpm_audit.log"):
        """
        Initialize military token manager

        Args:
            audit_log_path: Path to security audit log file
        """
        self.token_base_path = "/sys/devices/platform/dell-smbios.0/tokens"
        self.audit_log_path = audit_log_path
        self.token_cache = {}
        self.validation_cache = {}
        self.current_authorization_level = SecurityLevel.NONE
        self.audit_events = []
        self.session_id = None

        # Ensure audit log directory exists
        os.makedirs(os.path.dirname(audit_log_path), exist_ok=True)

        logger.info("Military Token Manager initialized")

    def validate_military_tokens(self, force_refresh: bool = False) -> TokenValidationResult:
        """
        Validate all required military tokens

        Args:
            force_refresh: Force refresh of cached token values

        Returns:
            TokenValidationResult with validation details
        """
        try:
            validation_timestamp = time.time()

            # Check cache first
            cache_key = "full_validation"
            if not force_refresh and cache_key in self.validation_cache:
                cached_result = self.validation_cache[cache_key]
                if validation_timestamp - cached_result.validation_timestamp < 300:  # 5 minute cache
                    logger.debug("Using cached token validation result")
                    return cached_result

            validation_result = TokenValidationResult(
                success=False,
                tokens_validated=[],
                tokens_missing=[],
                authorization_level=SecurityLevel.NONE,
                available_operations=[],
                security_warnings=[],
                validation_timestamp=validation_timestamp
            )

            # Validate each military token
            for token_id, token_info in self.MILITARY_TOKENS.items():
                token_result = self._validate_single_token(token_id, token_info)

                if token_result['success']:
                    validation_result.tokens_validated.append(token_result)
                    logger.info(f"Token {token_id} validated: {token_result['value']}")
                else:
                    validation_result.tokens_missing.append(token_id)
                    validation_result.security_warnings.append(token_result['error'])
                    logger.warning(f"Token {token_id} validation failed: {token_result['error']}")

            # Determine authorization level
            validated_count = len(validation_result.tokens_validated)
            validation_result.authorization_level = self._determine_authorization_level(validated_count)

            # Get available operations for authorization level
            if validation_result.authorization_level in self.SECURITY_LEVELS:
                validation_result.available_operations = self.SECURITY_LEVELS[
                    validation_result.authorization_level
                ]['tpm_operations']

            # Update success status
            validation_result.success = validated_count > 0

            # Update current authorization level
            self.current_authorization_level = validation_result.authorization_level

            # Cache successful validation
            self.validation_cache[cache_key] = validation_result

            # Log security event
            self._log_security_event(
                "TOKEN_VALIDATION",
                "validate_all_tokens",
                f"Validated {validated_count}/6 tokens, level: {validation_result.authorization_level.name}"
            )

            logger.info(f"Token validation complete: {validated_count}/6 tokens, level: {validation_result.authorization_level.name}")

            return validation_result

        except Exception as e:
            logger.error(f"Error during token validation: {e}")
            return TokenValidationResult(
                success=False,
                tokens_validated=[],
                tokens_missing=list(self.MILITARY_TOKENS.keys()),
                authorization_level=SecurityLevel.NONE,
                available_operations=[],
                security_warnings=[f"Validation error: {str(e)}"],
                validation_timestamp=time.time()
            )

    def validate_operation_authorization(self, operation: str,
                                       current_tokens: Optional[List[str]] = None) -> bool:
        """
        Validate if current tokens authorize the requested TPM operation

        Args:
            operation: TPM operation name
            current_tokens: Optional list of current token IDs (auto-detect if None)

        Returns:
            True if operation is authorized, False otherwise
        """
        try:
            # Get current tokens if not provided
            if current_tokens is None:
                validation_result = self.validate_military_tokens()
                current_tokens = [token['token_id'] for token in validation_result.tokens_validated]

            # Check if operation is in authorization matrix
            if operation not in self.TPM_AUTHORIZATION_MATRIX:
                self._log_security_event("AUTHORIZATION_DENIED", operation, "Unknown operation")
                logger.warning(f"Unknown operation: {operation}")
                return False

            operation_info = self.TPM_AUTHORIZATION_MATRIX[operation]
            required_tokens = set(operation_info['required_tokens'])
            available_tokens = set(current_tokens)

            # Check token requirements
            if not required_tokens.issubset(available_tokens):
                missing_tokens = required_tokens - available_tokens
                self._log_security_event(
                    "AUTHORIZATION_DENIED",
                    operation,
                    f"Missing required tokens: {list(missing_tokens)}"
                )
                logger.warning(f"Operation {operation} denied: missing tokens {list(missing_tokens)}")
                return False

            # Check security level requirements
            required_level = operation_info['security_level']
            current_level = self.current_authorization_level

            if current_level.value < required_level.value:
                self._log_security_event(
                    "AUTHORIZATION_DENIED",
                    operation,
                    f"Insufficient security level: {current_level.name} < {required_level.name}"
                )
                logger.warning(f"Operation {operation} denied: insufficient security level")
                return False

            # Authorization successful
            self._log_security_event(
                "AUTHORIZATION_GRANTED",
                operation,
                f"Authorized with {current_level.name} clearance"
            )
            logger.info(f"Operation {operation} authorized")
            return True

        except Exception as e:
            logger.error(f"Error validating operation authorization: {e}")
            self._log_security_event("AUTHORIZATION_ERROR", operation, f"Error: {str(e)}")
            return False

    def create_security_handshake(self, tokens_validated: List[Dict[str, Any]]) -> bytes:
        """
        Create security handshake data using validated military tokens

        Args:
            tokens_validated: List of validated token information

        Returns:
            Security handshake bytes for ME coordination
        """
        try:
            handshake_components = []

            # Add platform identifier
            platform_id = b'DELL_LAT5450_MILSPEC'
            handshake_components.append(platform_id)

            # Add validated tokens
            for token_info in tokens_validated:
                token_id = int(token_info['token_id'], 16)
                token_value = int(token_info['value'], 16)

                # Create token component
                token_component = struct.pack('>HI', token_id, token_value)
                handshake_components.append(token_component)

            # Add timestamp for freshness
            timestamp = int(time.time())
            timestamp_component = struct.pack('>Q', timestamp)
            handshake_components.append(timestamp_component)

            # Combine all components
            handshake_data = b''.join(handshake_components)

            # Add cryptographic hash for integrity
            handshake_hash = hashlib.sha256(handshake_data).digest()
            final_handshake = handshake_data + handshake_hash

            logger.info(f"Security handshake created: {len(final_handshake)} bytes")
            return final_handshake

        except Exception as e:
            logger.error(f"Error creating security handshake: {e}")
            return b''

    def get_current_authorization_info(self) -> Dict[str, Any]:
        """
        Get current authorization information

        Returns:
            Dictionary with current authorization details
        """
        return {
            'authorization_level': self.current_authorization_level.name,
            'level_value': self.current_authorization_level.value,
            'available_operations': self.SECURITY_LEVELS.get(
                self.current_authorization_level, {}
            ).get('tpm_operations', []),
            'encryption_capability': self.SECURITY_LEVELS.get(
                self.current_authorization_level, {}
            ).get('encryption', 'None'),
            'audit_level': self.SECURITY_LEVELS.get(
                self.current_authorization_level, {}
            ).get('audit_level', 'none')
        }

    def export_audit_log(self, output_path: Optional[str] = None) -> str:
        """
        Export security audit log for compliance reporting

        Args:
            output_path: Optional output file path

        Returns:
            Path to exported audit log
        """
        try:
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"/tmp/military_tpm_audit_export_{timestamp}.json"

            audit_export = {
                'export_timestamp': datetime.now(timezone.utc).isoformat(),
                'platform': 'Dell_Latitude_5450_MILSPEC',
                'audit_events': self.audit_events,
                'total_events': len(self.audit_events),
                'classification': 'UNCLASSIFIED // FOR OFFICIAL USE ONLY'
            }

            with open(output_path, 'w') as f:
                json.dump(audit_export, f, indent=2)

            logger.info(f"Audit log exported: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error exporting audit log: {e}")
            raise

    # Private helper methods

    def _validate_single_token(self, token_id: str, token_info: MilitaryTokenInfo) -> Dict[str, Any]:
        """Validate a single military token"""
        try:
            token_path = f"{self.token_base_path}/{token_id}_value"

            if not os.path.exists(token_path):
                return {
                    'success': False,
                    'token_id': token_id,
                    'error': f"Token file not found: {token_path}"
                }

            # Read token value
            with open(token_path, 'r') as f:
                value_str = f.read().strip()
                value = int(value_str, 16)

            # Validate token value range
            if value <= 0 or value == 0xFFFFFFFF:
                return {
                    'success': False,
                    'token_id': token_id,
                    'error': f"Invalid token value: 0x{value:08X}"
                }

            # Cache token value
            self.token_cache[token_id] = {
                'value': value,
                'last_read': time.time()
            }

            return {
                'success': True,
                'token_id': token_id,
                'name': token_info.name,
                'value': f"0x{value:08X}",
                'security_level': token_info.security_level.name,
                'status': 'VALID'
            }

        except Exception as e:
            return {
                'success': False,
                'token_id': token_id,
                'error': f"Token read error: {str(e)}"
            }

    def _determine_authorization_level(self, validated_count: int) -> SecurityLevel:
        """Determine authorization level based on validated token count"""
        if validated_count >= 6:
            return SecurityLevel.TOP_SECRET
        elif validated_count >= 4:
            return SecurityLevel.SECRET
        elif validated_count >= 2:
            return SecurityLevel.CONFIDENTIAL
        elif validated_count >= 1:
            return SecurityLevel.UNCLASSIFIED
        else:
            return SecurityLevel.NONE

    def _log_security_event(self, event_type: str, operation: str, details: str):
        """Log security event for military compliance"""
        try:
            event = SecurityAuditEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                session_id=self.session_id,
                event_type=event_type,
                operation=operation,
                details=details,
                platform='Dell_Latitude_5450_MILSPEC',
                authorization_level=self.current_authorization_level.name,
                tokens_used=list(self.token_cache.keys())
            )

            # Add to in-memory audit log
            self.audit_events.append(asdict(event))

            # Write to persistent audit log
            with open(self.audit_log_path, 'a') as f:
                f.write(json.dumps(asdict(event)) + '\n')

            logger.debug(f"Security event logged: {event_type} - {operation}")

        except Exception as e:
            logger.error(f"Error logging security event: {e}")

    def set_session_id(self, session_id: str):
        """Set session ID for audit logging"""
        self.session_id = session_id


# Convenience functions
def validate_military_tokens() -> TokenValidationResult:
    """Convenience function to validate military tokens"""
    manager = DellMilitaryTokenManager()
    return manager.validate_military_tokens()


def check_operation_authorization(operation: str) -> bool:
    """Convenience function to check operation authorization"""
    manager = DellMilitaryTokenManager()
    manager.validate_military_tokens()
    return manager.validate_operation_authorization(operation)


def get_authorization_level() -> str:
    """Convenience function to get current authorization level"""
    manager = DellMilitaryTokenManager()
    result = manager.validate_military_tokens()
    return result.authorization_level.name


if __name__ == "__main__":
    # Test the military token integration
    print("=== Military Token Integration Test ===")

    try:
        manager = DellMilitaryTokenManager()

        # Test token validation
        print("\n--- Token Validation ---")
        validation_result = manager.validate_military_tokens()

        print(f"Validation Success: {validation_result.success}")
        print(f"Authorization Level: {validation_result.authorization_level.name}")
        print(f"Tokens Validated: {len(validation_result.tokens_validated)}")
        print(f"Tokens Missing: {validation_result.tokens_missing}")
        print(f"Available Operations: {validation_result.available_operations}")

        if validation_result.security_warnings:
            print(f"Security Warnings: {validation_result.security_warnings}")

        # Test operation authorization
        print("\n--- Operation Authorization ---")
        test_operations = ['startup', 'pcrread', 'createkey', 'quote', 'nsa_algorithms']

        for operation in test_operations:
            authorized = manager.validate_operation_authorization(operation)
            print(f"{operation}: {'✓ AUTHORIZED' if authorized else '✗ DENIED'}")

        # Test security handshake creation
        print("\n--- Security Handshake ---")
        if validation_result.tokens_validated:
            handshake = manager.create_security_handshake(validation_result.tokens_validated)
            print(f"Handshake created: {len(handshake)} bytes")
        else:
            print("No tokens validated - cannot create handshake")

        # Show authorization info
        print("\n--- Authorization Info ---")
        auth_info = manager.get_current_authorization_info()
        for key, value in auth_info.items():
            print(f"{key}: {value}")

    except Exception as e:
        print(f"✗ Test error: {e}")