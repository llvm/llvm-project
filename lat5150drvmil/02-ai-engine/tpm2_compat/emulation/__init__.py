#!/usr/bin/env python3
"""
TPM2 Compatibility Layer Emulation Package
Device emulation components for transparent tpm2-tools compatibility

Author: C-INTERNAL Agent
Date: 2025-09-23
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
"""

from .device_emulator import (
    TPMDeviceEmulator,
    TPMDeviceInterceptor,
    EmulatedDevice,
    start_tpm_device_emulation
)

__all__ = [
    'TPMDeviceEmulator',
    'TPMDeviceInterceptor',
    'EmulatedDevice',
    'start_tpm_device_emulation'
]