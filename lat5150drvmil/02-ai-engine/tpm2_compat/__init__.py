#!/usr/bin/env python3
"""
TPM2 Compatibility Layer Package
Main package for TPM2 compatibility with ME-coordinated TPM implementations

Author: C-INTERNAL Agent
Date: 2025-09-23
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
"""

# Core components
from .core import (
    PCRAddressTranslator,
    MEInterfaceWrapper,
    DellMilitaryTokenManager,
    TPM2ProtocolBridge,
    create_tpm2_bridge,
    send_tpm_command_transparent
)

# Emulation components
from .emulation import (
    TPMDeviceEmulator,
    start_tpm_device_emulation
)

# Analysis tools
from .tools import (
    IntelNPUAnalyzer,
    analyze_npu_acceleration_potential
)

# Test framework
from .tests import run_compatibility_tests

# Package metadata
__version__ = "1.0.0"
__author__ = "C-INTERNAL Agent"
__description__ = "TPM2 Compatibility Layer for ME-coordinated TPM implementations"
__classification__ = "UNCLASSIFIED // FOR OFFICIAL USE ONLY"

# Main exports
__all__ = [
    # Core functionality
    'PCRAddressTranslator',
    'MEInterfaceWrapper',
    'DellMilitaryTokenManager',
    'TPM2ProtocolBridge',
    'create_tpm2_bridge',
    'send_tpm_command_transparent',

    # Device emulation
    'TPMDeviceEmulator',
    'start_tpm_device_emulation',

    # Analysis tools
    'IntelNPUAnalyzer',
    'analyze_npu_acceleration_potential',

    # Testing
    'run_compatibility_tests'
]