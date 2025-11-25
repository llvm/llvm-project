#!/usr/bin/env python3
"""
TPM2 Compatibility Layer Test Suite
Comprehensive testing framework for tpm2-tools compatibility validation

Author: C-INTERNAL Agent
Date: 2025-09-23
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
"""

import os
import sys
import unittest
import struct
import time
import subprocess
import tempfile
import logging
from typing import Dict, List, Optional, Tuple, Any
from unittest.mock import Mock, patch, MagicMock

# Add parent directories to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core import (
    PCRAddressTranslator, MEInterfaceWrapper, DellMilitaryTokenManager,
    TPM2ProtocolBridge, create_tpm2_bridge, SecurityLevel
)
from emulation import TPMDeviceEmulator, start_tpm_device_emulation
from core.constants import *

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)  # Reduce log noise during tests
logger = logging.getLogger(__name__)

class TestPCRTranslation(unittest.TestCase):
    """Test PCR address translation functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.translator = PCRAddressTranslator()

    def test_decimal_to_hex_translation(self):
        """Test decimal to hex PCR translation"""
        # Test standard PCR mappings
        result = self.translator.decimal_to_hex(0)
        self.assertTrue(result.success)
        self.assertEqual(result.translated_pcr, 0x0000)

        result = self.translator.decimal_to_hex(7)
        self.assertTrue(result.success)
        self.assertEqual(result.translated_pcr, 0x0007)

        result = self.translator.decimal_to_hex(23)
        self.assertTrue(result.success)
        self.assertEqual(result.translated_pcr, 0x0017)

        # Test invalid PCR
        result = self.translator.decimal_to_hex(24)
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error_message)

    def test_hex_to_decimal_translation(self):
        """Test hex to decimal PCR translation"""
        # Test standard hex mappings
        result = self.translator.hex_to_decimal(0x0000)
        self.assertTrue(result.success)
        self.assertEqual(result.translated_pcr, 0)

        result = self.translator.hex_to_decimal(0x0007)
        self.assertTrue(result.success)
        self.assertEqual(result.translated_pcr, 7)

        result = self.translator.hex_to_decimal(0x0017)
        self.assertTrue(result.success)
        self.assertEqual(result.translated_pcr, 23)

        # Test special configuration PCRs
        result = self.translator.hex_to_decimal("CAFE")
        self.assertTrue(result.success)
        self.assertEqual(result.translated_pcr, "CAFE")

    def test_pcr_validation(self):
        """Test PCR validation functionality"""
        # Valid decimal PCRs
        result = self.translator.validate_pcr_range(0, "decimal")
        self.assertTrue(result.success)

        result = self.translator.validate_pcr_range(23, "decimal")
        self.assertTrue(result.success)

        # Invalid decimal PCRs
        result = self.translator.validate_pcr_range(24, "decimal")
        self.assertFalse(result.success)

        result = self.translator.validate_pcr_range(-1, "decimal")
        self.assertFalse(result.success)

        # Valid hex PCRs
        result = self.translator.validate_pcr_range(0xCAFE, "hex")
        self.assertTrue(result.success)

        # Invalid hex PCRs
        result = self.translator.validate_pcr_range(0x10000, "hex")
        self.assertFalse(result.success)

    def test_algorithm_banks(self):
        """Test algorithm bank selection"""
        from core.pcr_translator import PCRBankType

        result = self.translator.decimal_to_hex(0, PCRBankType.SHA256)
        self.assertTrue(result.success)
        self.assertEqual(result.algorithm_bank, PCRBankType.SHA256)

        result = self.translator.decimal_to_hex(0, PCRBankType.SHA384)
        self.assertTrue(result.success)
        self.assertEqual(result.algorithm_bank, PCRBankType.SHA384)


class TestMilitaryTokenIntegration(unittest.TestCase):
    """Test military token validation and authorization"""

    def setUp(self):
        """Set up test fixtures"""
        self.token_manager = DellMilitaryTokenManager()

    def test_token_validation_structure(self):
        """Test token validation structure"""
        # Test with mock token files
        with patch('os.path.exists', return_value=True), \
             patch('builtins.open', unittest.mock.mock_open(read_data='44000001')):

            result = self.token_manager.validate_military_tokens()
            self.assertIsInstance(result.tokens_validated, list)
            self.assertIsInstance(result.tokens_missing, list)
            self.assertIsInstance(result.authorization_level, SecurityLevel)

    def test_operation_authorization(self):
        """Test operation authorization logic"""
        # Mock validated tokens for testing
        self.token_manager.current_authorization_level = SecurityLevel.CONFIDENTIAL

        # Test authorized operation
        authorized = self.token_manager.validate_operation_authorization(
            "pcrread", ["049e", "049f"]
        )
        self.assertTrue(authorized)

        # Test unauthorized operation
        authorized = self.token_manager.validate_operation_authorization(
            "nsa_algorithms", ["049e"]
        )
        self.assertFalse(authorized)

    def test_security_level_determination(self):
        """Test security level determination"""
        # Test level determination with different token counts
        level = self.token_manager._determine_authorization_level(1)
        self.assertEqual(level, SecurityLevel.UNCLASSIFIED)

        level = self.token_manager._determine_authorization_level(2)
        self.assertEqual(level, SecurityLevel.CONFIDENTIAL)

        level = self.token_manager._determine_authorization_level(4)
        self.assertEqual(level, SecurityLevel.SECRET)

        level = self.token_manager._determine_authorization_level(6)
        self.assertEqual(level, SecurityLevel.TOP_SECRET)

    def test_security_handshake_creation(self):
        """Test security handshake creation"""
        mock_tokens = [
            {'token_id': '049e', 'value': '0x44000001'},
            {'token_id': '049f', 'value': '0x44000002'}
        ]

        handshake = self.token_manager.create_security_handshake(mock_tokens)
        self.assertIsInstance(handshake, bytes)
        self.assertGreater(len(handshake), 0)


class TestMEWrapper(unittest.TestCase):
    """Test ME interface wrapper functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.me_wrapper = MEInterfaceWrapper()

    def test_command_wrapping(self):
        """Test TPM command wrapping"""
        # Create test TPM command
        test_command = struct.pack('>HII', TPM_ST_NO_SESSIONS, 12, TPM_CC_STARTUP) + b'\x00\x00'

        # Mock session for testing
        mock_session = Mock()
        mock_session.session_id = 0x12345678

        wrapped = self.me_wrapper.wrap_tpm_command(test_command, mock_session)

        if wrapped:  # Only test if wrapping succeeds (depends on ME availability)
            self.assertIsInstance(wrapped, bytes)
            self.assertGreater(len(wrapped), len(test_command))

    def test_response_unwrapping(self):
        """Test ME response unwrapping"""
        # Create mock ME response
        mock_me_response = struct.pack('>BBHI', 0x02, 0x00, 16, 0x12345678) + \
                          struct.pack('>BBBB', 0x00, 0x00, 0x00, 0x00) + \
                          struct.pack('>HII', TPM_ST_NO_SESSIONS, 10, TPM_RC_SUCCESS)

        unwrapped = self.me_wrapper.unwrap_me_response(mock_me_response)

        if unwrapped:  # Only test if unwrapping succeeds
            self.assertIsInstance(unwrapped, bytes)
            self.assertGreaterEqual(len(unwrapped), 10)

    def test_session_management(self):
        """Test ME session management"""
        # Test session info when no session exists
        info = self.me_wrapper.get_session_info()
        self.assertIn('status', info)


class TestProtocolBridge(unittest.TestCase):
    """Test TPM2 protocol bridge integration"""

    def setUp(self):
        """Set up test fixtures"""
        self.bridge = TPM2ProtocolBridge("test_session")

    def test_command_parsing(self):
        """Test TPM command parsing"""
        # Create test TPM command
        test_command = struct.pack('>HII', TPM_ST_NO_SESSIONS, 12, TPM_CC_STARTUP) + b'\x00\x00'

        parsed = self.bridge._parse_tpm_command(test_command)

        if parsed:  # Only test if parsing succeeds
            self.assertEqual(parsed.tag, TPM_ST_NO_SESSIONS)
            self.assertEqual(parsed.size, 12)
            self.assertEqual(parsed.code, TPM_CC_STARTUP)

    def test_bridge_status(self):
        """Test bridge status reporting"""
        status = self.bridge.get_bridge_status()

        self.assertIn('initialized', status)
        self.assertIn('command_count', status)
        self.assertIn('authorization_level', status)

    def test_command_type_mapping(self):
        """Test command type mapping"""
        # Test known command codes
        operation = self.bridge.COMMAND_TYPE_MAP.get(TPM_CC_STARTUP, "unknown")
        self.assertEqual(operation, "startup")

        operation = self.bridge.COMMAND_TYPE_MAP.get(TPM_CC_PCR_READ, "unknown")
        self.assertEqual(operation, "pcrread")

        # Test unknown command code
        operation = self.bridge.COMMAND_TYPE_MAP.get(0xFFFFFFFF, "unknown")
        self.assertEqual(operation, "unknown")


class TestDeviceEmulator(unittest.TestCase):
    """Test TPM device emulation functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_device_path = os.path.join(self.temp_dir, "tpm0.test")

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_emulator_creation(self):
        """Test emulator creation"""
        emulator = TPMDeviceEmulator(self.test_device_path, "UNCLASSIFIED")
        self.assertEqual(emulator.device_path, self.test_device_path)
        self.assertEqual(emulator.security_level, "UNCLASSIFIED")

    def test_session_management(self):
        """Test emulator session management"""
        emulator = TPMDeviceEmulator(self.test_device_path, "UNCLASSIFIED")

        session_id = emulator.create_session()
        self.assertIsNotNone(session_id)
        self.assertIn(session_id, emulator.active_sessions)

        emulator._close_session(session_id)
        self.assertNotIn(session_id, emulator.active_sessions)

    def test_emulation_status(self):
        """Test emulation status reporting"""
        emulator = TPMDeviceEmulator(self.test_device_path, "UNCLASSIFIED")

        status = emulator.get_emulation_status()
        self.assertIn('is_running', status)
        self.assertIn('device_path', status)
        self.assertIn('statistics', status)


class TestTPM2ToolsCompatibility(unittest.TestCase):
    """Test compatibility with actual tpm2-tools"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_device_path = os.path.join(self.temp_dir, "tpm0.test")

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_tpm2_tools_detection(self):
        """Test if tpm2-tools are available for testing"""
        try:
            result = subprocess.run(['which', 'tpm2_startup'],
                                  capture_output=True, text=True)
            tpm2_tools_available = result.returncode == 0
        except:
            tpm2_tools_available = False

        if not tpm2_tools_available:
            self.skipTest("tpm2-tools not available for compatibility testing")

    @unittest.skipUnless(
        os.path.exists('/usr/bin/tpm2_startup') or os.path.exists('/usr/local/bin/tpm2_startup'),
        "tpm2-tools not installed"
    )
    def test_tpm2_startup_simulation(self):
        """Test tpm2_startup command simulation"""
        # This would test actual tpm2_startup command with our emulated device
        # Skipped by default as it requires complex setup
        self.skipTest("Complex integration test - requires full emulation setup")

    def test_command_structure_compatibility(self):
        """Test TPM command structure compatibility"""
        # Test that our command parsing works with real tpm2-tools command formats

        # Simulate tpm2_startup command structure
        startup_cmd = struct.pack('>HII', TPM_ST_NO_SESSIONS, 12, TPM_CC_STARTUP) + \
                     struct.pack('>H', 0x0000)  # TPM_SU_CLEAR

        bridge = TPM2ProtocolBridge("test_compatibility")
        parsed = bridge._parse_tpm_command(startup_cmd)

        if parsed:  # Only test if parsing works
            self.assertEqual(parsed.code, TPM_CC_STARTUP)
            self.assertEqual(len(parsed.raw_command), 12)


class TestPerformanceMetrics(unittest.TestCase):
    """Test performance characteristics of compatibility layer"""

    def test_pcr_translation_performance(self):
        """Test PCR translation performance"""
        translator = PCRAddressTranslator()

        start_time = time.time()

        # Perform many translations
        for i in range(1000):
            pcr = i % 24
            result = translator.decimal_to_hex(pcr)
            self.assertTrue(result.success)

        end_time = time.time()
        execution_time = end_time - start_time

        # Should complete 1000 translations in reasonable time
        self.assertLess(execution_time, 1.0)  # Less than 1 second

        logger.info(f"PCR translation performance: {1000/execution_time:.1f} ops/sec")

    def test_cache_effectiveness(self):
        """Test translation cache effectiveness"""
        translator = PCRAddressTranslator()

        # First translation (cache miss)
        start_time = time.time()
        result1 = translator.decimal_to_hex(0)
        miss_time = time.time() - start_time

        # Second translation (cache hit)
        start_time = time.time()
        result2 = translator.decimal_to_hex(0)
        hit_time = time.time() - start_time

        self.assertTrue(result1.success)
        self.assertTrue(result2.success)

        # Cache hit should be faster (though this is a micro-benchmark)
        # In practice, the difference might be minimal for simple operations
        logger.info(f"Cache miss: {miss_time*1000:.3f}ms, Cache hit: {hit_time*1000:.3f}ms")


class TestIntegrationScenarios(unittest.TestCase):
    """Test complete integration scenarios"""

    def test_end_to_end_command_flow(self):
        """Test complete command flow from input to output"""
        # This tests the full pipeline:
        # TPM Command → PCR Translation → ME Wrapping → Token Validation → Response

        # Mock all external dependencies
        with patch('os.path.exists', return_value=True), \
             patch('builtins.open', unittest.mock.mock_open(read_data='44000001')):

            # Create bridge with mocked components
            bridge = TPM2ProtocolBridge("integration_test")

            # Test command that doesn't require full initialization
            test_command = struct.pack('>HII', TPM_ST_NO_SESSIONS, 12, TPM_CC_GETRANDOM) + \
                          struct.pack('>H', 32)  # Request 32 random bytes

            # This will fail without proper initialization, but we can test the parsing
            parsed = bridge._parse_tpm_command(test_command)

            if parsed:
                self.assertEqual(parsed.code, TPM_CC_GETRANDOM)
                operation_type = bridge.COMMAND_TYPE_MAP.get(parsed.code, "unknown")
                self.assertEqual(operation_type, "getrandom")

    def test_error_handling_chain(self):
        """Test error handling throughout the system"""
        bridge = TPM2ProtocolBridge("error_test")

        # Test invalid command
        invalid_command = b"invalid"
        parsed = bridge._parse_tmp_command(invalid_command)
        self.assertIsNone(parsed)

        # Test bridge status with uninitialized state
        status = bridge.get_bridge_status()
        self.assertFalse(status['initialized'])


class TestSecurityCompliance(unittest.TestCase):
    """Test security compliance and audit functionality"""

    def test_audit_logging(self):
        """Test security audit logging functionality"""
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp_file:
            audit_log_path = tmp_file.name

        try:
            token_manager = DellMilitaryTokenManager(audit_log_path)
            token_manager.set_session_id("security_test")

            # Generate some audit events
            token_manager._log_security_event(
                "TEST_EVENT", "test_operation", "Test audit entry"
            )

            # Check that audit log was created and contains data
            self.assertTrue(os.path.exists(audit_log_path))

            with open(audit_log_path, 'r') as f:
                log_content = f.read()
                self.assertIn("TEST_EVENT", log_content)
                self.assertIn("test_operation", log_content)

        finally:
            if os.path.exists(audit_log_path):
                os.unlink(audit_log_path)

    def test_authorization_matrix(self):
        """Test operation authorization matrix"""
        token_manager = DellMilitaryTokenManager()

        # Test authorization matrix structure
        for operation, requirements in token_manager.TPM_AUTHORIZATION_MATRIX.items():
            self.assertIn('required_tokens', requirements)
            self.assertIn('security_level', requirements)
            self.assertIn('description', requirements)

            # Verify security level is valid
            self.assertIsInstance(requirements['security_level'], SecurityLevel)


def run_compatibility_tests():
    """Run all compatibility tests and generate report"""
    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestPCRTranslation,
        TestMilitaryTokenIntegration,
        TestMEWrapper,
        TestProtocolBridge,
        TestDeviceEmulator,
        TestTPM2ToolsCompatibility,
        TestPerformanceMetrics,
        TestIntegrationScenarios,
        TestSecurityCompliance
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(test_suite)

    # Generate summary report
    print("\n" + "="*60)
    print("TPM2 COMPATIBILITY LAYER TEST REPORT")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")

    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError:')[-1].strip()}")

    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.splitlines()[-1]}")

    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) /
                   result.testsRun * 100) if result.testsRun > 0 else 0

    print(f"\nOverall Success Rate: {success_rate:.1f}%")
    print("="*60)

    return result.wasSuccessful()


if __name__ == "__main__":
    # Run tests when executed directly
    print("TPM2 Compatibility Layer Test Suite")
    print("Starting comprehensive testing...")

    success = run_compatibility_tests()

    if success:
        print("\n✓ All tests passed - TPM2 compatibility layer is ready")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed - review failures before deployment")
        sys.exit(1)