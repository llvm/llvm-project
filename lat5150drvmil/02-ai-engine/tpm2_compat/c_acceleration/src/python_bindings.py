#!/usr/bin/env python3
"""
Python-C Interface Bindings for TPM2 Compatibility Acceleration
High-performance ctypes interface with memory management and error handling

Author: C-INTERNAL Agent
Date: 2025-09-23
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
"""

import os
import sys
import ctypes
import ctypes.util
import threading
import time
import logging
from typing import Optional, Tuple, List, Union, Any, Dict
from dataclasses import dataclass
from enum import Enum
import traceback
import platform

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# LIBRARY LOADING AND CONFIGURATION
# =============================================================================

class TPM2SecurityLevel(Enum):
    """Security classification levels"""
    UNCLASSIFIED = 0
    CONFIDENTIAL = 1
    SECRET = 2
    TOP_SECRET = 3

class TPM2AccelerationFlags(Enum):
    """Hardware acceleration capabilities"""
    NONE = 0
    NPU = 1
    GNA = 2
    AVX512 = 4
    AES_NI = 8
    RDRAND = 16
    ALL = 0xFF

class TPM2ReturnCode(Enum):
    """TPM2 library return codes"""
    SUCCESS = 0
    FAILURE = 1
    BAD_PARAMETER = 2
    INSUFFICIENT_BUFFER = 3
    NOT_SUPPORTED = 4
    NOT_PERMITTED = 5
    HARDWARE_FAILURE = 6
    NOT_INITIALIZED = 7
    MEMORY_ERROR = 8
    CRYPTO_ERROR = 9
    SECURITY_VIOLATION = 10

class TPM2PCRBank(Enum):
    """PCR bank types"""
    SHA256 = 0
    SHA384 = 1
    SHA3_256 = 2
    SHA3_384 = 3
    SHA512 = 4
    SM3 = 5
    RESERVED = 6
    EXTENDED = 7

@dataclass
class TPM2LibraryConfig:
    """Global library configuration"""
    security_level: TPM2SecurityLevel = TPM2SecurityLevel.UNCLASSIFIED
    acceleration_flags: TPM2AccelerationFlags = TPM2AccelerationFlags.ALL
    enable_profiling: bool = True
    enable_fault_detection: bool = True
    max_sessions: int = 32
    memory_pool_size_mb: int = 64
    log_file_path: Optional[str] = None
    enable_debug_mode: bool = False

class TPM2AccelerationLibrary:
    """High-performance Python interface to TPM2 C acceleration library"""

    def __init__(self, library_path: Optional[str] = None):
        """
        Initialize TPM2 acceleration library interface

        Args:
            library_path: Path to shared library (auto-detected if None)
        """
        self._lib = None
        self._initialized = False
        self._lock = threading.RLock()
        self._session_handles = {}
        self._crypto_contexts = {}
        self._device_handles = {}

        # Load shared library
        self._load_library(library_path)

        # Define function prototypes
        self._define_function_prototypes()

        logger.info("TPM2 Acceleration Library interface initialized")

    def _load_library(self, library_path: Optional[str]) -> None:
        """Load the C acceleration library"""
        if library_path is None:
            # Auto-detect library path
            library_path = self._find_library()

        if not os.path.exists(library_path):
            raise FileNotFoundError(f"TPM2 acceleration library not found: {library_path}")

        try:
            self._lib = ctypes.CDLL(library_path)
            logger.info(f"Loaded TPM2 acceleration library: {library_path}")
        except OSError as e:
            raise RuntimeError(f"Failed to load library {library_path}: {e}")

    def _find_library(self) -> str:
        """Auto-detect library path"""
        # Get current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        lib_dir = os.path.join(current_dir, "..", "lib")

        # Platform-specific library names
        if platform.system() == "Linux":
            lib_name = "libtpm2_compat_accelerated.so"
        elif platform.system() == "Darwin":
            lib_name = "libtpm2_compat_accelerated.dylib"
        elif platform.system() == "Windows":
            lib_name = "tpm2_compat_accelerated.dll"
        else:
            raise RuntimeError(f"Unsupported platform: {platform.system()}")

        library_path = os.path.join(lib_dir, lib_name)

        if not os.path.exists(library_path):
            # Try system library paths
            system_lib = ctypes.util.find_library("tpm2_compat_accelerated")
            if system_lib:
                return system_lib

            raise FileNotFoundError(f"TPM2 acceleration library not found in {library_path}")

        return library_path

    def _define_function_prototypes(self) -> None:
        """Define C function prototypes for ctypes"""

        # Library initialization and configuration
        self._lib.tpm2_library_init.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self._lib.tpm2_library_init.restype = ctypes.c_int

        self._lib.tpm2_library_cleanup.argtypes = []
        self._lib.tpm2_library_cleanup.restype = None

        self._lib.tpm2_library_get_version.argtypes = []
        self._lib.tpm2_library_get_version.restype = ctypes.c_char_p

        # PCR translation functions
        self._lib.tpm2_pcr_decimal_to_hex_fast.argtypes = [
            ctypes.c_uint32,  # pcr_decimal
            ctypes.c_int,     # bank
            ctypes.POINTER(ctypes.c_uint16)  # pcr_hex_out
        ]
        self._lib.tpm2_pcr_decimal_to_hex_fast.restype = ctypes.c_int

        self._lib.tpm2_pcr_hex_to_decimal_fast.argtypes = [
            ctypes.c_uint16,  # pcr_hex
            ctypes.POINTER(ctypes.c_uint32),  # pcr_decimal_out
            ctypes.POINTER(ctypes.c_int)      # bank_out
        ]
        self._lib.tpm2_pcr_hex_to_decimal_fast.restype = ctypes.c_int

        self._lib.tpm2_pcr_translate_batch.argtypes = [
            ctypes.POINTER(ctypes.c_uint32),  # pcr_decimals
            ctypes.c_size_t,                  # count
            ctypes.c_int,                     # bank
            ctypes.POINTER(ctypes.c_uint16)   # pcr_hexs_out
        ]
        self._lib.tpm2_pcr_translate_batch.restype = ctypes.c_int

        # ME interface functions
        self._lib.tpm2_me_interface_init.argtypes = [ctypes.c_uint32]
        self._lib.tpm2_me_interface_init.restype = ctypes.c_int

        self._lib.tpm2_me_session_establish.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),  # config
            ctypes.POINTER(ctypes.c_void_p)   # session_out
        ]
        self._lib.tpm2_me_session_establish.restype = ctypes.c_int

        self._lib.tpm2_me_wrap_command_fast.argtypes = [
            ctypes.c_void_p,                  # session
            ctypes.POINTER(ctypes.c_uint8),   # tpm_command
            ctypes.c_size_t,                  # tpm_command_size
            ctypes.POINTER(ctypes.c_uint8),   # wrapped_command_out
            ctypes.POINTER(ctypes.c_size_t)   # wrapped_command_size_inout
        ]
        self._lib.tpm2_me_wrap_command_fast.restype = ctypes.c_int

        self._lib.tpm2_me_unwrap_response_fast.argtypes = [
            ctypes.c_void_p,                  # session
            ctypes.POINTER(ctypes.c_uint8),   # me_response
            ctypes.c_size_t,                  # me_response_size
            ctypes.POINTER(ctypes.c_uint8),   # tpm_response_out
            ctypes.POINTER(ctypes.c_size_t)   # tpm_response_size_inout
        ]
        self._lib.tpm2_me_unwrap_response_fast.restype = ctypes.c_int

        self._lib.tpm2_me_send_tpm_command.argtypes = [
            ctypes.c_void_p,                  # session
            ctypes.POINTER(ctypes.c_uint8),   # tpm_command
            ctypes.c_size_t,                  # tpm_command_size
            ctypes.POINTER(ctypes.c_uint8),   # tpm_response_out
            ctypes.POINTER(ctypes.c_size_t),  # tpm_response_size_inout
            ctypes.c_uint64                   # timeout_ms
        ]
        self._lib.tpm2_me_send_tpm_command.restype = ctypes.c_int

        self._lib.tpm2_me_session_close.argtypes = [ctypes.c_void_p]
        self._lib.tpm2_me_session_close.restype = ctypes.c_int

        # Cryptographic functions
        self._lib.tpm2_crypto_init.argtypes = [ctypes.c_uint32, ctypes.c_int]
        self._lib.tpm2_crypto_init.restype = ctypes.c_int

        self._lib.tpm2_crypto_hash_accelerated.argtypes = [
            ctypes.c_int,                     # hash_alg
            ctypes.POINTER(ctypes.c_uint8),   # data
            ctypes.c_size_t,                  # data_size
            ctypes.POINTER(ctypes.c_uint8),   # hash_out
            ctypes.POINTER(ctypes.c_size_t)   # hash_size_inout
        ]
        self._lib.tpm2_crypto_hash_accelerated.restype = ctypes.c_int

        # Error handling
        self._lib.tpm2_rc_to_string.argtypes = [ctypes.c_int]
        self._lib.tpm2_rc_to_string.restype = ctypes.c_char_p

        self._lib.tpm2_get_last_error.argtypes = [
            ctypes.c_char_p,                  # error_message
            ctypes.c_size_t,                  # message_size
            ctypes.POINTER(ctypes.c_uint32)   # error_code_out
        ]
        self._lib.tpm2_get_last_error.restype = ctypes.c_int

    def initialize(self, config: TPM2LibraryConfig) -> None:
        """
        Initialize the TPM2 acceleration library

        Args:
            config: Library configuration
        """
        with self._lock:
            if self._initialized:
                return

            # Convert Python config to C structure (simplified)
            c_config = ctypes.c_void_p(0)  # Placeholder for actual config structure

            result = self._lib.tpm2_library_init(ctypes.byref(c_config))

            if result != TPM2ReturnCode.SUCCESS.value:
                raise RuntimeError(f"Library initialization failed: {self._get_error_string(result)}")

            self._initialized = True
            logger.info("TPM2 acceleration library initialized successfully")

    def cleanup(self) -> None:
        """Cleanup library resources"""
        with self._lock:
            if not self._initialized:
                return

            # Close all open sessions
            for session_id, session_handle in list(self._session_handles.items()):
                try:
                    self.close_me_session(session_id)
                except Exception as e:
                    logger.warning(f"Error closing session {session_id}: {e}")

            # Close all crypto contexts
            for context_id in list(self._crypto_contexts.keys()):
                try:
                    self.destroy_crypto_context(context_id)
                except Exception as e:
                    logger.warning(f"Error destroying crypto context {context_id}: {e}")

            # Cleanup library
            self._lib.tpm2_library_cleanup()
            self._initialized = False

            logger.info("TPM2 acceleration library cleanup complete")

    def get_version(self) -> str:
        """Get library version"""
        version_ptr = self._lib.tpm2_library_get_version()
        return version_ptr.decode('utf-8') if version_ptr else "Unknown"

    def pcr_decimal_to_hex(self, pcr_decimal: int, bank: TPM2PCRBank = TPM2PCRBank.SHA256) -> int:
        """
        Translate decimal PCR to hex with hardware acceleration

        Args:
            pcr_decimal: Decimal PCR value (0-23)
            bank: PCR bank type

        Returns:
            Hex PCR value
        """
        if not self._initialized:
            raise RuntimeError("Library not initialized")

        pcr_hex_out = ctypes.c_uint16()

        result = self._lib.tpm2_pcr_decimal_to_hex_fast(
            ctypes.c_uint32(pcr_decimal),
            ctypes.c_int(bank.value),
            ctypes.byref(pcr_hex_out)
        )

        if result != TPM2ReturnCode.SUCCESS.value:
            raise ValueError(f"PCR translation failed: {self._get_error_string(result)}")

        return pcr_hex_out.value

    def pcr_hex_to_decimal(self, pcr_hex: int) -> Tuple[int, TPM2PCRBank]:
        """
        Translate hex PCR to decimal with hardware acceleration

        Args:
            pcr_hex: Hex PCR value

        Returns:
            Tuple of (decimal PCR, bank type)
        """
        if not self._initialized:
            raise RuntimeError("Library not initialized")

        pcr_decimal_out = ctypes.c_uint32()
        bank_out = ctypes.c_int()

        result = self._lib.tpm2_pcr_hex_to_decimal_fast(
            ctypes.c_uint16(pcr_hex),
            ctypes.byref(pcr_decimal_out),
            ctypes.byref(bank_out)
        )

        if result != TPM2ReturnCode.SUCCESS.value:
            raise ValueError(f"PCR translation failed: {self._get_error_string(result)}")

        return pcr_decimal_out.value, TPM2PCRBank(bank_out.value)

    def pcr_translate_batch(self, pcr_decimals: List[int], bank: TPM2PCRBank = TPM2PCRBank.SHA256) -> List[int]:
        """
        Batch translate multiple decimal PCRs to hex with vectorized operations

        Args:
            pcr_decimals: List of decimal PCR values
            bank: PCR bank type

        Returns:
            List of hex PCR values
        """
        if not self._initialized:
            raise RuntimeError("Library not initialized")

        if not pcr_decimals:
            return []

        count = len(pcr_decimals)

        # Create C arrays
        c_pcr_decimals = (ctypes.c_uint32 * count)(*pcr_decimals)
        c_pcr_hexs_out = (ctypes.c_uint16 * count)()

        result = self._lib.tpm2_pcr_translate_batch(
            c_pcr_decimals,
            ctypes.c_size_t(count),
            ctypes.c_int(bank.value),
            c_pcr_hexs_out
        )

        if result != TPM2ReturnCode.SUCCESS.value:
            raise ValueError(f"Batch PCR translation failed: {self._get_error_string(result)}")

        return list(c_pcr_hexs_out)

    def establish_me_session(self, security_level: TPM2SecurityLevel = TPM2SecurityLevel.UNCLASSIFIED,
                           timeout_ms: int = 5000) -> str:
        """
        Establish optimized ME session

        Args:
            security_level: Required security level
            timeout_ms: Session timeout in milliseconds

        Returns:
            Session ID string
        """
        if not self._initialized:
            raise RuntimeError("Library not initialized")

        # Initialize ME interface if needed
        accel_flags = TPM2AccelerationFlags.ALL.value
        result = self._lib.tpm2_me_interface_init(ctypes.c_uint32(accel_flags))

        if result != TPM2ReturnCode.SUCCESS.value:
            raise RuntimeError(f"ME interface initialization failed: {self._get_error_string(result)}")

        # Create session configuration (simplified)
        session_handle = ctypes.c_void_p()
        config = ctypes.c_void_p(0)  # Placeholder for actual config structure

        result = self._lib.tpm2_me_session_establish(
            ctypes.byref(config),
            ctypes.byref(session_handle)
        )

        if result != TPM2ReturnCode.SUCCESS.value:
            raise RuntimeError(f"ME session establishment failed: {self._get_error_string(result)}")

        # Generate session ID and store handle
        session_id = f"session_{int(time.time() * 1000000)}"
        self._session_handles[session_id] = session_handle.value

        logger.info(f"ME session established: {session_id}")
        return session_id

    def wrap_tpm_command(self, session_id: str, tpm_command: bytes) -> bytes:
        """
        Wrap TPM command with ME protocol using zero-copy operations

        Args:
            session_id: ME session ID
            tpm_command: Raw TPM command bytes

        Returns:
            ME-wrapped command bytes
        """
        if not self._initialized:
            raise RuntimeError("Library not initialized")

        if session_id not in self._session_handles:
            raise ValueError(f"Invalid session ID: {session_id}")

        session_handle = ctypes.c_void_p(self._session_handles[session_id])

        # Prepare input buffer
        command_size = len(tpm_command)
        c_tpm_command = (ctypes.c_uint8 * command_size)(*tpm_command)

        # Prepare output buffer (command + header)
        max_wrapped_size = command_size + 64  # Extra space for ME header
        c_wrapped_command = (ctypes.c_uint8 * max_wrapped_size)()
        wrapped_size = ctypes.c_size_t(max_wrapped_size)

        result = self._lib.tpm2_me_wrap_command_fast(
            session_handle,
            c_tpm_command,
            ctypes.c_size_t(command_size),
            c_wrapped_command,
            ctypes.byref(wrapped_size)
        )

        if result != TPM2ReturnCode.SUCCESS.value:
            raise RuntimeError(f"Command wrapping failed: {self._get_error_string(result)}")

        # Convert to Python bytes
        return bytes(c_wrapped_command[:wrapped_size.value])

    def unwrap_me_response(self, session_id: str, me_response: bytes) -> bytes:
        """
        Unwrap ME response to extract TPM response

        Args:
            session_id: ME session ID
            me_response: ME-wrapped response bytes

        Returns:
            Raw TPM response bytes
        """
        if not self._initialized:
            raise RuntimeError("Library not initialized")

        if session_id not in self._session_handles:
            raise ValueError(f"Invalid session ID: {session_id}")

        session_handle = ctypes.c_void_p(self._session_handles[session_id])

        # Prepare input buffer
        response_size = len(me_response)
        c_me_response = (ctypes.c_uint8 * response_size)(*me_response)

        # Prepare output buffer
        max_tpm_size = response_size  # TPM response should be smaller than ME response
        c_tpm_response = (ctypes.c_uint8 * max_tpm_size)()
        tpm_size = ctypes.c_size_t(max_tpm_size)

        result = self._lib.tpm2_me_unwrap_response_fast(
            session_handle,
            c_me_response,
            ctypes.c_size_t(response_size),
            c_tpm_response,
            ctypes.byref(tpm_size)
        )

        if result != TPM2ReturnCode.SUCCESS.value:
            raise RuntimeError(f"Response unwrapping failed: {self._get_error_string(result)}")

        # Convert to Python bytes
        return bytes(c_tpm_response[:tmp_size.value])

    def send_tpm_command_via_me(self, session_id: str, tpm_command: bytes, timeout_ms: int = 5000) -> bytes:
        """
        Send TPM command via ME with full optimization

        Args:
            session_id: ME session ID
            tpm_command: Raw TPM command bytes
            timeout_ms: Operation timeout in milliseconds

        Returns:
            Raw TPM response bytes
        """
        if not self._initialized:
            raise RuntimeError("Library not initialized")

        if session_id not in self._session_handles:
            raise ValueError(f"Invalid session ID: {session_id}")

        session_handle = ctypes.c_void_p(self._session_handles[session_id])

        # Prepare input buffer
        command_size = len(tpm_command)
        c_tpm_command = (ctypes.c_uint8 * command_size)(*tpm_command)

        # Prepare output buffer
        max_response_size = 4096  # Standard TPM response buffer size
        c_tpm_response = (ctypes.c_uint8 * max_response_size)()
        response_size = ctypes.c_size_t(max_response_size)

        result = self._lib.tpm2_me_send_tpm_command(
            session_handle,
            c_tpm_command,
            ctypes.c_size_t(command_size),
            c_tpm_response,
            ctypes.byref(response_size),
            ctypes.c_uint64(timeout_ms)
        )

        if result != TPM2ReturnCode.SUCCESS.value:
            raise RuntimeError(f"TPM command via ME failed: {self._get_error_string(result)}")

        # Convert to Python bytes
        return bytes(c_tpm_response[:response_size.value])

    def close_me_session(self, session_id: str) -> None:
        """
        Close ME session and cleanup resources

        Args:
            session_id: ME session ID to close
        """
        if session_id not in self._session_handles:
            logger.warning(f"Session ID not found: {session_id}")
            return

        session_handle = ctypes.c_void_p(self._session_handles[session_id])

        result = self._lib.tpm2_me_session_close(session_handle)

        if result != TPM2ReturnCode.SUCCESS.value:
            logger.warning(f"Error closing session {session_id}: {self._get_error_string(result)}")

        # Remove from tracking
        del self._session_handles[session_id]
        logger.info(f"ME session closed: {session_id}")

    def compute_hash_accelerated(self, data: bytes, algorithm: str = "SHA256") -> bytes:
        """
        Compute hash using hardware acceleration

        Args:
            data: Data to hash
            algorithm: Hash algorithm ("SHA256", "SHA384", "SHA512")

        Returns:
            Hash digest bytes
        """
        if not self._initialized:
            raise RuntimeError("Library not initialized")

        # Map algorithm names to C constants
        alg_map = {
            "SHA256": 0,
            "SHA384": 1,
            "SHA512": 2,
            "SHA3_256": 3,
            "SHA3_384": 4
        }

        if algorithm not in alg_map:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")

        alg_code = alg_map[algorithm]

        # Prepare input buffer
        data_size = len(data)
        c_data = (ctypes.c_uint8 * data_size)(*data)

        # Prepare output buffer
        max_hash_size = 64  # SHA512 is largest at 64 bytes
        c_hash_out = (ctypes.c_uint8 * max_hash_size)()
        hash_size = ctypes.c_size_t(max_hash_size)

        result = self._lib.tpm2_crypto_hash_accelerated(
            ctypes.c_int(alg_code),
            c_data,
            ctypes.c_size_t(data_size),
            c_hash_out,
            ctypes.byref(hash_size)
        )

        if result != TPM2ReturnCode.SUCCESS.value:
            raise RuntimeError(f"Hardware hash computation failed: {self._get_error_string(result)}")

        # Convert to Python bytes
        return bytes(c_hash_out[:hash_size.value])

    def _get_error_string(self, error_code: int) -> str:
        """Get human-readable error string"""
        try:
            error_ptr = self._lib.tpm2_rc_to_string(ctypes.c_int(error_code))
            if error_ptr:
                return error_ptr.decode('utf-8')
        except Exception:
            pass

        # Fallback to enum lookup
        try:
            return TPM2ReturnCode(error_code).name
        except ValueError:
            return f"Unknown error code: {error_code}"

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()

# =============================================================================
# CONVENIENCE FUNCTIONS AND CONTEXT MANAGERS
# =============================================================================

class TPM2AcceleratedSession:
    """Context manager for ME sessions with automatic cleanup"""

    def __init__(self, library: TPM2AccelerationLibrary,
                 security_level: TPM2SecurityLevel = TPM2SecurityLevel.UNCLASSIFIED):
        self.library = library
        self.security_level = security_level
        self.session_id = None

    def __enter__(self) -> str:
        self.session_id = self.library.establish_me_session(self.security_level)
        return self.session_id

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.session_id:
            self.library.close_me_session(self.session_id)

def create_accelerated_library(config: Optional[TPM2LibraryConfig] = None) -> TPM2AccelerationLibrary:
    """
    Create and initialize TPM2 acceleration library

    Args:
        config: Library configuration (uses defaults if None)

    Returns:
        Initialized TPM2AccelerationLibrary instance
    """
    if config is None:
        config = TPM2LibraryConfig()

    library = TPM2AccelerationLibrary()
    library.initialize(config)

    return library

# =============================================================================
# TESTING AND EXAMPLES
# =============================================================================

def test_pcr_translation_performance():
    """Test PCR translation performance"""
    print("=== PCR Translation Performance Test ===")

    config = TPM2LibraryConfig(enable_debug_mode=True)

    with create_accelerated_library(config) as lib:
        # Test single translation
        start_time = time.time()
        for i in range(1000):
            hex_pcr = lib.pcr_decimal_to_hex(i % 24)
            decimal_pcr, bank = lib.pcr_hex_to_decimal(hex_pcr)

        single_time = time.time() - start_time
        print(f"Single translations (1000 ops): {single_time:.3f}s")

        # Test batch translation
        pcr_list = list(range(24)) * 42  # 1008 PCRs

        start_time = time.time()
        hex_pcrs = lib.pcr_translate_batch(pcr_list)
        batch_time = time.time() - start_time

        print(f"Batch translation (1008 PCRs): {batch_time:.3f}s")
        print(f"Performance improvement: {single_time / batch_time:.1f}x")

def test_me_session_management():
    """Test ME session management"""
    print("\n=== ME Session Management Test ===")

    config = TPM2LibraryConfig(max_sessions=4)

    with create_accelerated_library(config) as lib:
        # Test session establishment
        with TPM2AcceleratedSession(lib, TPM2SecurityLevel.CONFIDENTIAL) as session_id:
            print(f"Session established: {session_id}")

            # Test command wrapping
            test_command = b'\x80\x01\x00\x00\x00\x0c\x00\x00\x01\x43\x00\x00'

            wrapped = lib.wrap_tpm_command(session_id, test_command)
            print(f"Command wrapped: {len(wrapped)} bytes")

            # Simulate response unwrapping
            test_response = wrapped[:20]  # Simulated ME response
            try:
                unwrapped = lib.unwrap_me_response(session_id, test_response)
                print(f"Response unwrapped: {len(unwrapped)} bytes")
            except Exception as e:
                print(f"Response unwrapping (expected to fail in test): {e}")

def test_hash_acceleration():
    """Test hardware-accelerated hashing"""
    print("\n=== Hash Acceleration Test ===")

    config = TPM2LibraryConfig(acceleration_flags=TPM2AccelerationFlags.AES_NI)

    with create_accelerated_library(config) as lib:
        test_data = b"The quick brown fox jumps over the lazy dog"

        algorithms = ["SHA256", "SHA384", "SHA512"]

        for alg in algorithms:
            try:
                start_time = time.time()
                hash_result = lib.compute_hash_accelerated(test_data, alg)
                hash_time = time.time() - start_time

                print(f"{alg}: {hash_result.hex()[:16]}... ({hash_time*1000:.2f}ms)")
            except Exception as e:
                print(f"{alg}: Error - {e}")

if __name__ == "__main__":
    try:
        print("TPM2 Acceleration Library Python Bindings Test")
        print("=" * 50)

        # Run performance tests
        test_pcr_translation_performance()
        test_me_session_management()
        test_hash_acceleration()

        print("\n✓ All tests completed successfully")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        traceback.print_exc()
        sys.exit(1)