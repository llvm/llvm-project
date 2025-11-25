#!/usr/bin/env python3
"""
PCR Address Translator for TPM2 Compatibility Layer
Translates between standard decimal PCR addressing (0-23) and extended hex range (0x0000-0xFFFF)

Author: C-INTERNAL Agent
Date: 2025-09-23
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PCRBankType(Enum):
    """PCR bank types for algorithm selection"""
    SHA256 = 0
    SHA384 = 1
    SHA3_256 = 2
    SHA3_384 = 3
    SHA512 = 4
    SM3 = 5
    RESERVED = 6
    EXTENDED = 7

@dataclass
class PCRTranslationResult:
    """Result of PCR address translation"""
    success: bool
    translated_pcr: Optional[Union[int, str]]
    algorithm_bank: Optional[PCRBankType]
    error_message: Optional[str] = None
    original_pcr: Optional[Union[int, str]] = None

class PCRAddressTranslator:
    """
    Core PCR address translator for ME-TPM compatibility
    Handles bidirectional translation between standard and extended PCR addressing
    """

    # Standard PCR to Extended Hex mapping (0-23 → 0x0000-0xFFFF)
    STANDARD_TO_HEX_MAP = {
        # BIOS/UEFI PCRs (0-7)
        0: 0x0000,   # BIOS measurements
        1: 0x0001,   # BIOS configuration
        2: 0x0002,   # Option ROM Code
        3: 0x0003,   # Option ROM Configuration
        4: 0x0004,   # IPL (Master Boot Record)
        5: 0x0005,   # IPL Configuration
        6: 0x0006,   # State Transition/Wake
        7: 0x0007,   # Platform Manufacturer

        # OS PCRs (8-15)
        8: 0x0008,   # OS Loader
        9: 0x0009,   # OS Configuration
        10: 0x000A,  # IMA Template
        11: 0x000B,  # Kernel Command Line
        12: 0x000C,  # Kernel Modules
        13: 0x000D,  # OS Boot
        14: 0x000E,  # MokList
        15: 0x000F,  # System Boot

        # Extended PCRs (16-23)
        16: 0x0010,  # Debug
        17: 0x0011,  # Dynamic Root of Trust
        18: 0x0012,  # Trusted OS
        19: 0x0013,  # Trusted OS Configuration
        20: 0x0014,  # Trusted OS Data
        21: 0x0015,  # OS Applications
        22: 0x0016,  # OS Application Configuration
        23: 0x0017,  # OS Application Data
    }

    # Special Configuration PCRs
    SPECIAL_CONFIG_PCRS = {
        'CAFE': 0xCAFE,  # Algorithm configuration
        'BEEF': 0xBEEF,  # Extended functionality
        'DEAD': 0xDEAD,  # Debug/diagnostic
        'FACE': 0xFACE,  # Factory configuration
    }

    # Reverse mapping for hex → decimal translation
    HEX_TO_STANDARD_MAP = {v: k for k, v in STANDARD_TO_HEX_MAP.items()}

    # PCR bank mappings for extended addressing
    BANK_SELECTORS = {
        0: PCRBankType.SHA256,
        1: PCRBankType.SHA384,
        2: PCRBankType.SHA3_256,
        3: PCRBankType.SHA3_384,
        4: PCRBankType.SHA512,
        5: PCRBankType.SM3,
        6: PCRBankType.RESERVED,
        7: PCRBankType.EXTENDED
    }

    def __init__(self):
        """Initialize PCR address translator"""
        self.translation_cache = {}
        logger.info("PCR Address Translator initialized")

    def decimal_to_hex(self, pcr_decimal: int,
                      algorithm_bank: Optional[PCRBankType] = None) -> PCRTranslationResult:
        """
        Convert standard decimal PCR (0-23) to hex PCR (0x0000-0xFFFF)

        Args:
            pcr_decimal: Standard PCR index (0-23)
            algorithm_bank: Optional algorithm bank for extended addressing

        Returns:
            PCRTranslationResult with translated hex PCR or error
        """
        try:
            # Validate input range
            if not isinstance(pcr_decimal, int):
                return PCRTranslationResult(
                    success=False,
                    translated_pcr=None,
                    algorithm_bank=None,
                    error_message=f"PCR must be integer, got {type(pcr_decimal)}",
                    original_pcr=pcr_decimal
                )

            if pcr_decimal < 0 or pcr_decimal > 23:
                return PCRTranslationResult(
                    success=False,
                    translated_pcr=None,
                    algorithm_bank=None,
                    error_message=f"PCR {pcr_decimal} out of standard range (0-23)",
                    original_pcr=pcr_decimal
                )

            # Check cache first
            cache_key = f"d2h_{pcr_decimal}_{algorithm_bank}"
            if cache_key in self.translation_cache:
                cached_result = self.translation_cache[cache_key]
                logger.debug(f"Cache hit for decimal PCR {pcr_decimal}")
                return cached_result

            # Direct mapping for standard PCRs
            if pcr_decimal in self.STANDARD_TO_HEX_MAP:
                hex_pcr = self.STANDARD_TO_HEX_MAP[pcr_decimal]

                # Apply algorithm bank offset if specified
                if algorithm_bank and algorithm_bank != PCRBankType.SHA256:
                    bank_offset = list(self.BANK_SELECTORS.keys())[
                        list(self.BANK_SELECTORS.values()).index(algorithm_bank)
                    ]
                    hex_pcr = hex_pcr | (bank_offset << 8)

                result = PCRTranslationResult(
                    success=True,
                    translated_pcr=hex_pcr,
                    algorithm_bank=algorithm_bank or PCRBankType.SHA256,
                    original_pcr=pcr_decimal
                )

                # Cache successful translation
                self.translation_cache[cache_key] = result

                logger.info(f"Translated decimal PCR {pcr_decimal} → 0x{hex_pcr:04X}")
                return result

            # Fallback for extended range
            return PCRTranslationResult(
                success=False,
                translated_pcr=None,
                algorithm_bank=None,
                error_message=f"No mapping found for decimal PCR {pcr_decimal}",
                original_pcr=pcr_decimal
            )

        except Exception as e:
            logger.error(f"Error translating decimal PCR {pcr_decimal}: {e}")
            return PCRTranslationResult(
                success=False,
                translated_pcr=None,
                algorithm_bank=None,
                error_message=f"Translation error: {str(e)}",
                original_pcr=pcr_decimal
            )

    def hex_to_decimal(self, pcr_hex: Union[int, str]) -> PCRTranslationResult:
        """
        Convert hex PCR (0x0000-0xFFFF) to standard decimal PCR (0-23)

        Args:
            pcr_hex: Hex PCR value (int or hex string)

        Returns:
            PCRTranslationResult with translated decimal PCR or error
        """
        try:
            # Normalize hex input
            if isinstance(pcr_hex, str):
                if pcr_hex.upper() in self.SPECIAL_CONFIG_PCRS:
                    pcr_hex_val = self.SPECIAL_CONFIG_PCRS[pcr_hex.upper()]
                else:
                    pcr_hex_val = int(pcr_hex, 16) if pcr_hex.startswith('0x') else int(pcr_hex, 16)
            else:
                pcr_hex_val = pcr_hex

            # Validate hex range
            if pcr_hex_val < 0 or pcr_hex_val > 0xFFFF:
                return PCRTranslationResult(
                    success=False,
                    translated_pcr=None,
                    algorithm_bank=None,
                    error_message=f"Hex PCR 0x{pcr_hex_val:04X} out of extended range (0x0000-0xFFFF)",
                    original_pcr=pcr_hex
                )

            # Check cache first
            cache_key = f"h2d_{pcr_hex_val:04X}"
            if cache_key in self.translation_cache:
                cached_result = self.translation_cache[cache_key]
                logger.debug(f"Cache hit for hex PCR 0x{pcr_hex_val:04X}")
                return cached_result

            # Extract bank selector and base PCR
            base_pcr = pcr_hex_val & 0xFF
            bank_selector = (pcr_hex_val >> 8) & 0xFF

            # Determine algorithm bank
            algorithm_bank = self.BANK_SELECTORS.get(bank_selector & 0x7, PCRBankType.SHA256)

            # Handle special configuration PCRs
            if pcr_hex_val in [0xCAFE, 0xBEEF, 0xDEAD, 0xFACE]:
                special_name = next(
                    name for name, value in self.SPECIAL_CONFIG_PCRS.items()
                    if value == pcr_hex_val
                )
                result = PCRTranslationResult(
                    success=True,
                    translated_pcr=special_name,
                    algorithm_bank=PCRBankType.EXTENDED,
                    original_pcr=pcr_hex
                )

                # Cache successful translation
                self.translation_cache[cache_key] = result

                logger.info(f"Translated hex PCR 0x{pcr_hex_val:04X} → {special_name} (config)")
                return result

            # Direct reverse mapping for standard PCRs
            if pcr_hex_val in self.HEX_TO_STANDARD_MAP:
                decimal_pcr = self.HEX_TO_STANDARD_MAP[pcr_hex_val]

                result = PCRTranslationResult(
                    success=True,
                    translated_pcr=decimal_pcr,
                    algorithm_bank=algorithm_bank,
                    original_pcr=pcr_hex
                )

                # Cache successful translation
                self.translation_cache[cache_key] = result

                logger.info(f"Translated hex PCR 0x{pcr_hex_val:04X} → {decimal_pcr}")
                return result

            # Extended range mapping
            if base_pcr <= 23:
                result = PCRTranslationResult(
                    success=True,
                    translated_pcr=base_pcr,
                    algorithm_bank=algorithm_bank,
                    original_pcr=pcr_hex
                )

                # Cache successful translation
                self.translation_cache[cache_key] = result

                logger.info(f"Translated extended hex PCR 0x{pcr_hex_val:04X} → {base_pcr} (bank {algorithm_bank.name})")
                return result

            # No mapping found
            return PCRTranslationResult(
                success=False,
                translated_pcr=None,
                algorithm_bank=None,
                error_message=f"No mapping found for hex PCR 0x{pcr_hex_val:04X}",
                original_pcr=pcr_hex
            )

        except Exception as e:
            logger.error(f"Error translating hex PCR {pcr_hex}: {e}")
            return PCRTranslationResult(
                success=False,
                translated_pcr=None,
                algorithm_bank=None,
                error_message=f"Translation error: {str(e)}",
                original_pcr=pcr_hex
            )

    def validate_pcr_range(self, pcr: Union[int, str],
                          pcr_type: str = "auto") -> PCRTranslationResult:
        """
        Validate PCR is in supported range (decimal 0-23 or hex 0x0000-0xFFFF)

        Args:
            pcr: PCR value to validate
            pcr_type: "decimal", "hex", or "auto" for auto-detection

        Returns:
            PCRTranslationResult with validation result
        """
        try:
            # Auto-detect PCR type
            if pcr_type == "auto":
                if isinstance(pcr, str):
                    if pcr.upper() in self.SPECIAL_CONFIG_PCRS:
                        pcr_type = "hex"
                    elif pcr.startswith('0x') or len(pcr) > 2:
                        pcr_type = "hex"
                    else:
                        pcr_type = "decimal"
                else:
                    pcr_type = "decimal" if pcr <= 23 else "hex"

            if pcr_type == "decimal":
                if isinstance(pcr, str):
                    pcr_val = int(pcr)
                else:
                    pcr_val = pcr

                if 0 <= pcr_val <= 23:
                    return PCRTranslationResult(
                        success=True,
                        translated_pcr=pcr_val,
                        algorithm_bank=PCRBankType.SHA256,
                        original_pcr=pcr
                    )
                else:
                    return PCRTranslationResult(
                        success=False,
                        translated_pcr=None,
                        algorithm_bank=None,
                        error_message=f"Decimal PCR {pcr_val} out of range (0-23)",
                        original_pcr=pcr
                    )

            elif pcr_type == "hex":
                if isinstance(pcr, str):
                    if pcr.upper() in self.SPECIAL_CONFIG_PCRS:
                        pcr_val = self.SPECIAL_CONFIG_PCRS[pcr.upper()]
                    else:
                        pcr_val = int(pcr, 16) if pcr.startswith('0x') else int(pcr, 16)
                else:
                    pcr_val = pcr

                if 0 <= pcr_val <= 0xFFFF:
                    return PCRTranslationResult(
                        success=True,
                        translated_pcr=pcr_val,
                        algorithm_bank=PCRBankType.SHA256,
                        original_pcr=pcr
                    )
                else:
                    return PCRTranslationResult(
                        success=False,
                        translated_pcr=None,
                        algorithm_bank=None,
                        error_message=f"Hex PCR 0x{pcr_val:04X} out of range (0x0000-0xFFFF)",
                        original_pcr=pcr
                    )

            return PCRTranslationResult(
                success=False,
                translated_pcr=None,
                algorithm_bank=None,
                error_message=f"Unknown PCR type: {pcr_type}",
                original_pcr=pcr
            )

        except Exception as e:
            logger.error(f"Error validating PCR {pcr}: {e}")
            return PCRTranslationResult(
                success=False,
                translated_pcr=None,
                algorithm_bank=None,
                error_message=f"Validation error: {str(e)}",
                original_pcr=pcr
            )

    def get_supported_pcrs(self) -> Dict[str, List]:
        """
        Get all supported PCR ranges and special configurations

        Returns:
            Dictionary with supported PCR information
        """
        return {
            'standard_decimal': list(range(24)),  # 0-23
            'standard_hex': [f"0x{i:04X}" for i in self.STANDARD_TO_HEX_MAP.values()],
            'special_config': list(self.SPECIAL_CONFIG_PCRS.keys()),
            'algorithm_banks': [bank.name for bank in PCRBankType],
            'extended_range': "0x0000-0xFFFF"
        }

    def clear_cache(self):
        """Clear translation cache"""
        cache_size = len(self.translation_cache)
        self.translation_cache.clear()
        logger.info(f"Cleared translation cache ({cache_size} entries)")


# Convenience functions for easy integration
def translate_decimal_to_hex(pcr: int, algorithm_bank: Optional[str] = None) -> Tuple[bool, Union[int, str]]:
    """
    Simple function to translate decimal PCR to hex

    Args:
        pcr: Decimal PCR (0-23)
        algorithm_bank: Optional algorithm bank name

    Returns:
        Tuple of (success, hex_pcr_or_error_message)
    """
    translator = PCRAddressTranslator()

    bank = None
    if algorithm_bank:
        try:
            bank = PCRBankType[algorithm_bank.upper()]
        except KeyError:
            return False, f"Unknown algorithm bank: {algorithm_bank}"

    result = translator.decimal_to_hex(pcr, bank)

    if result.success:
        return True, result.translated_pcr
    else:
        return False, result.error_message


def translate_hex_to_decimal(pcr_hex: Union[int, str]) -> Tuple[bool, Union[int, str]]:
    """
    Simple function to translate hex PCR to decimal

    Args:
        pcr_hex: Hex PCR value

    Returns:
        Tuple of (success, decimal_pcr_or_error_message)
    """
    translator = PCRAddressTranslator()
    result = translator.hex_to_decimal(pcr_hex)

    if result.success:
        return True, result.translated_pcr
    else:
        return False, result.error_message


if __name__ == "__main__":
    # Test the PCR translator
    translator = PCRAddressTranslator()

    print("=== PCR Address Translator Test ===")

    # Test decimal to hex translation
    print("\n--- Decimal to Hex Translation ---")
    for pcr in [0, 7, 16, 23]:
        result = translator.decimal_to_hex(pcr)
        if result.success:
            print(f"PCR {pcr} → 0x{result.translated_pcr:04X}")
        else:
            print(f"PCR {pcr} → ERROR: {result.error_message}")

    # Test hex to decimal translation
    print("\n--- Hex to Decimal Translation ---")
    for pcr_hex in [0x0000, 0x0007, 0xCAFE, 0xBEEF]:
        result = translator.hex_to_decimal(pcr_hex)
        if result.success:
            print(f"0x{pcr_hex:04X} → {result.translated_pcr}")
        else:
            print(f"0x{pcr_hex:04X} → ERROR: {result.error_message}")

    # Test special configuration PCRs
    print("\n--- Special Configuration PCRs ---")
    for name in ["CAFE", "BEEF", "DEAD", "FACE"]:
        result = translator.hex_to_decimal(name)
        if result.success:
            print(f"{name} → {result.translated_pcr} (0x{translator.SPECIAL_CONFIG_PCRS[name]:04X})")
        else:
            print(f"{name} → ERROR: {result.error_message}")

    # Show supported PCRs
    print("\n--- Supported PCR Information ---")
    supported = translator.get_supported_pcrs()
    for category, values in supported.items():
        print(f"{category}: {values}")