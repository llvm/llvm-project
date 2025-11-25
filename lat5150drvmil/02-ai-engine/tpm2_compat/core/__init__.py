#!/usr/bin/env python3
"""
TPM2 Compatibility Layer Core Package
Core components for TPM2 compatibility with ME-coordinated TPM implementation

Author: C-INTERNAL Agent
Date: 2025-09-23
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
"""

# Core component imports
from .pcr_translator import (
    PCRAddressTranslator,
    PCRTranslationResult,
    PCRBankType,
    translate_decimal_to_hex,
    translate_hex_to_decimal
)

from .me_wrapper import (
    MEInterfaceWrapper,
    MESessionContext,
    MECommandType,
    MEStatusCode,
    create_me_wrapper,
    send_tpm_command_with_me
)

from .military_token_integration import (
    DellMilitaryTokenManager,
    SecurityLevel,
    TokenValidationResult,
    SecurityAuditEvent,
    validate_military_tokens,
    check_operation_authorization,
    get_authorization_level
)

from .protocol_bridge import (
    TPM2ProtocolBridge,
    BridgeOperationResult,
    BridgeResult,
    create_tpm2_bridge,
    send_tpm_command_transparent
)

# Package metadata
__version__ = "1.0.0"
__author__ = "C-INTERNAL Agent"
__description__ = "TPM2 Compatibility Layer for ME-coordinated TPM implementations"
__classification__ = "UNCLASSIFIED // FOR OFFICIAL USE ONLY"

# Export all public components
__all__ = [
    # PCR Translator
    'PCRAddressTranslator',
    'PCRTranslationResult',
    'PCRBankType',
    'translate_decimal_to_hex',
    'translate_hex_to_decimal',

    # ME Wrapper
    'MEInterfaceWrapper',
    'MESessionContext',
    'MECommandType',
    'MEStatusCode',
    'create_me_wrapper',
    'send_tpm_command_with_me',

    # Military Token Integration
    'DellMilitaryTokenManager',
    'SecurityLevel',
    'TokenValidationResult',
    'SecurityAuditEvent',
    'validate_military_tokens',
    'check_operation_authorization',
    'get_authorization_level',

    # Protocol Bridge
    'TPM2ProtocolBridge',
    'BridgeOperationResult',
    'BridgeResult',
    'create_tpm2_bridge',
    'send_tpm_command_transparent'
]