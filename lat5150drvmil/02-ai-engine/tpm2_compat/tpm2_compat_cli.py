#!/usr/bin/env python3
"""
TPM2 Compatibility Layer CLI Tool
Command-line interface for managing and testing TPM2 compatibility layer

Author: C-INTERNAL Agent
Date: 2025-09-23
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
"""

import os
import sys
import argparse
import json
import time
import signal
import logging
from typing import Dict, List, Optional, Any

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from core import (
    create_tpm2_bridge, validate_military_tokens, get_authorization_level,
    translate_decimal_to_hex, translate_hex_to_decimal
)
from emulation import start_tpm_device_emulation
from tools import analyze_npu_acceleration_potential
from tests import run_compatibility_tests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TPM2CompatCLI:
    """Command-line interface for TPM2 compatibility layer"""

    def __init__(self):
        """Initialize CLI"""
        self.running_services = {}
        self.setup_signal_handlers()

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\nReceived signal {signum}, shutting down...")
        self.cleanup()
        sys.exit(0)

    def cleanup(self):
        """Cleanup running services"""
        for service_name, service in self.running_services.items():
            try:
                if hasattr(service, 'cleanup'):
                    service.cleanup()
                elif hasattr(service, 'stop_emulation'):
                    service.stop_emulation()
                print(f"✓ Stopped {service_name}")
            except Exception as e:
                print(f"✗ Error stopping {service_name}: {e}")

    def cmd_status(self, args):
        """Show system status and compatibility layer information"""
        print("=== TPM2 Compatibility Layer Status ===\n")

        try:
            # Military token validation
            print("--- Military Token Status ---")
            validation_result = validate_military_tokens()

            print(f"Authorization Level: {validation_result.authorization_level.name}")
            print(f"Tokens Validated: {len(validation_result.tokens_validated)}/6")
            print(f"Available Operations: {len(validation_result.available_operations)}")

            if validation_result.security_warnings:
                print("Security Warnings:")
                for warning in validation_result.security_warnings:
                    print(f"  ⚠ {warning}")

            # NPU analysis
            print("\n--- NPU Acceleration Status ---")
            try:
                npu_result = analyze_npu_acceleration_potential()
                print(f"NPU Hardware: {'✓ Detected' if npu_result.hardware_detected else '✗ Not detected'}")
                print(f"Capability Level: {npu_result.capability_level.name}")
                print(f"Recommended Algorithms: {len(npu_result.recommended_algorithms)}")
            except Exception as e:
                print(f"NPU Analysis Error: {e}")

            # System information
            print("\n--- System Information ---")
            print(f"Platform: {os.uname().sysname} {os.uname().release}")
            print(f"Python Version: {sys.version.split()[0]}")

            print("\n✓ Status check complete")

        except Exception as e:
            print(f"✗ Error getting status: {e}")
            return 1

        return 0

    def cmd_test_tokens(self, args):
        """Test military token validation"""
        print("=== Military Token Validation Test ===\n")

        try:
            validation_result = validate_military_tokens()

            print(f"Validation Success: {validation_result.success}")
            print(f"Authorization Level: {validation_result.authorization_level.name}")

            print(f"\nValidated Tokens ({len(validation_result.tokens_validated)}):")
            for token in validation_result.tokens_validated:
                print(f"  ✓ {token['token_id']}: {token['value']} ({token['security_level']})")

            if validation_result.tokens_missing:
                print(f"\nMissing Tokens ({len(validation_result.tokens_missing)}):")
                for token_id in validation_result.tokens_missing:
                    print(f"  ✗ {token_id}")

            if validation_result.security_warnings:
                print(f"\nSecurity Warnings:")
                for warning in validation_result.security_warnings:
                    print(f"  ⚠ {warning}")

            print(f"\nAvailable Operations: {validation_result.available_operations}")

            return 0 if validation_result.success else 1

        except Exception as e:
            print(f"✗ Token validation error: {e}")
            return 1

    def cmd_test_pcr(self, args):
        """Test PCR address translation"""
        print("=== PCR Address Translation Test ===\n")

        try:
            # Test decimal to hex
            print("--- Decimal to Hex Translation ---")
            test_pcrs = [0, 7, 16, 23]

            for pcr in test_pcrs:
                success, result = translate_decimal_to_hex(pcr)
                if success:
                    print(f"PCR {pcr} → 0x{result:04X}")
                else:
                    print(f"PCR {pcr} → ERROR: {result}")

            # Test hex to decimal
            print("\n--- Hex to Decimal Translation ---")
            test_hex_pcrs = [0x0000, 0x0007, 0x0010, 0xCAFE]

            for pcr_hex in test_hex_pcrs:
                success, result = translate_hex_to_decimal(pcr_hex)
                if success:
                    print(f"0x{pcr_hex:04X} → {result}")
                else:
                    print(f"0x{pcr_hex:04X} → ERROR: {result}")

            # Test special configuration PCRs
            print("\n--- Special Configuration PCRs ---")
            special_pcrs = ["CAFE", "BEEF", "DEAD", "FACE"]

            for pcr_name in special_pcrs:
                success, result = translate_hex_to_decimal(pcr_name)
                if success:
                    print(f"{pcr_name} → {result}")
                else:
                    print(f"{pcr_name} → ERROR: {result}")

            print("\n✓ PCR translation test complete")
            return 0

        except Exception as e:
            print(f"✗ PCR translation test error: {e}")
            return 1

    def cmd_start_emulation(self, args):
        """Start TPM device emulation"""
        print(f"=== Starting TPM Device Emulation ===\n")

        try:
            device_path = args.device_path or "/dev/tpm0.compat"
            security_level = args.security_level or "UNCLASSIFIED"

            print(f"Device Path: {device_path}")
            print(f"Security Level: {security_level}")

            emulator = start_tpm_device_emulation(device_path, security_level)

            if emulator:
                self.running_services['emulator'] = emulator
                print(f"✓ TPM device emulation started: {device_path}")

                if args.daemon:
                    print("Running in daemon mode... (Ctrl+C to stop)")
                    try:
                        while True:
                            time.sleep(1)
                    except KeyboardInterrupt:
                        pass
                else:
                    print("Emulation started successfully")

                return 0
            else:
                print("✗ Failed to start TPM device emulation")
                return 1

        except Exception as e:
            print(f"✗ Emulation error: {e}")
            return 1

    def cmd_run_tests(self, args):
        """Run compatibility test suite"""
        print("=== Running TPM2 Compatibility Tests ===\n")

        try:
            success = run_compatibility_tests()
            return 0 if success else 1

        except Exception as e:
            print(f"✗ Test execution error: {e}")
            return 1

    def cmd_analyze_npu(self, args):
        """Analyze NPU acceleration potential"""
        print("=== NPU Acceleration Analysis ===\n")

        try:
            result = analyze_npu_acceleration_potential()

            print(f"Hardware Detected: {'✓' if result.hardware_detected else '✗'}")
            print(f"Capability Level: {result.capability_level.name}")

            if result.npu_specifications:
                specs = result.npu_specifications
                print(f"\nNPU Specifications:")
                print(f"  Model: {specs.model}")
                print(f"  Performance: {specs.total_tops} TOPS")
                print(f"  GNA Version: {specs.gna_version}")

            print(f"\nRecommended Algorithms ({len(result.recommended_algorithms)}):")
            for alg in result.recommended_algorithms:
                print(f"  • {alg.value}")

            print(f"\nPerformance Projections:")
            for metric, value in result.performance_projections.items():
                print(f"  {metric}: {value:.2f}")

            if args.export:
                from tools.npu_acceleration_analysis import IntelNPUAnalyzer
                analyzer = IntelNPUAnalyzer()
                report_path = analyzer.export_analysis_report(result, args.export)
                print(f"\n✓ Analysis report exported: {report_path}")

            return 0

        except Exception as e:
            print(f"✗ NPU analysis error: {e}")
            return 1

    def cmd_bridge_test(self, args):
        """Test protocol bridge functionality"""
        print("=== Protocol Bridge Test ===\n")

        try:
            print("Creating protocol bridge...")
            bridge = create_tpm2_bridge("cli_test", args.security_level or "UNCLASSIFIED")

            if bridge:
                print("✓ Protocol bridge created successfully")

                # Show bridge status
                status = bridge.get_bridge_status()
                print(f"Authorization Level: {status['authorization_level']}")
                print(f"ME Session Active: {status['me_session_active']}")
                print(f"Tokens Validated: {status['tokens_validated']}")

                # Cleanup
                bridge.cleanup()
                print("✓ Bridge test complete")
                return 0
            else:
                print("✗ Failed to create protocol bridge")
                return 1

        except Exception as e:
            print(f"✗ Bridge test error: {e}")
            return 1

    def cmd_info(self, args):
        """Show compatibility layer information"""
        print("=== TPM2 Compatibility Layer Information ===\n")

        print("DESCRIPTION:")
        print("  TPM2 Compatibility Layer for ME-coordinated TPM implementations")
        print("  Enables standard tpm2-tools to work with non-standard hex PCR addressing")
        print("  and Intel ME command wrapping while maintaining military-grade security.")

        print("\nCOMPONENTS:")
        print("  • PCR Address Translator: Decimal ↔ Hex PCR translation (0-23 ↔ 0x0000-0xFFFF)")
        print("  • ME Command Wrapper: Intel ME protocol wrapping with session management")
        print("  • Military Token Integration: Dell military token validation (6 levels)")
        print("  • Protocol Bridge: TPM2 ↔ ME communication coordination")
        print("  • Device Emulator: /dev/tpm0 compatibility interface")
        print("  • NPU Acceleration: Intel Core Ultra 7 165H NPU optimization")

        print("\nSECURITY LEVELS:")
        print("  • UNCLASSIFIED: Basic TPM operations")
        print("  • CONFIDENTIAL: Crypto operations and key management")
        print("  • SECRET: Advanced crypto and attestation")
        print("  • TOP_SECRET: Quantum crypto and NSA algorithms")

        print("\nCLASSIFICATION: UNCLASSIFIED // FOR OFFICIAL USE ONLY")
        print("AUTHOR: C-INTERNAL Agent")
        print("VERSION: 1.0.0")

        return 0

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="TPM2 Compatibility Layer CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s status                    # Show system status
  %(prog)s test-tokens              # Test military token validation
  %(prog)s test-pcr                 # Test PCR address translation
  %(prog)s start-emulation --daemon # Start device emulation in background
  %(prog)s run-tests                # Run compatibility test suite
  %(prog)s analyze-npu --export report.json  # Analyze NPU and export report
  %(prog)s bridge-test --security-level SECRET  # Test protocol bridge

Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
        """
    )

    # Global options
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--log-file', type=str,
                       help='Log file path')

    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Status command
    status_parser = subparsers.add_parser('status', help='Show system status')

    # Token testing
    token_parser = subparsers.add_parser('test-tokens', help='Test military token validation')

    # PCR testing
    pcr_parser = subparsers.add_parser('test-pcr', help='Test PCR address translation')

    # Device emulation
    emulation_parser = subparsers.add_parser('start-emulation', help='Start TPM device emulation')
    emulation_parser.add_argument('--device-path', type=str,
                                 help='Device path for emulation (default: /dev/tpm0.compat)')
    emulation_parser.add_argument('--security-level', type=str,
                                 choices=['UNCLASSIFIED', 'CONFIDENTIAL', 'SECRET', 'TOP_SECRET'],
                                 help='Security level for operations')
    emulation_parser.add_argument('--daemon', action='store_true',
                                 help='Run in daemon mode')

    # Test suite
    test_parser = subparsers.add_parser('run-tests', help='Run compatibility test suite')

    # NPU analysis
    npu_parser = subparsers.add_parser('analyze-npu', help='Analyze NPU acceleration potential')
    npu_parser.add_argument('--export', type=str,
                           help='Export analysis report to file')

    # Bridge testing
    bridge_parser = subparsers.add_parser('bridge-test', help='Test protocol bridge')
    bridge_parser.add_argument('--security-level', type=str,
                              choices=['UNCLASSIFIED', 'CONFIDENTIAL', 'SECRET', 'TOP_SECRET'],
                              help='Security level for bridge test')

    # Info command
    info_parser = subparsers.add_parser('info', help='Show compatibility layer information')

    # Parse arguments
    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logging.getLogger().addHandler(file_handler)

    # Create CLI instance
    cli = TPM2CompatCLI()

    # Execute command
    try:
        if args.command == 'status':
            return cli.cmd_status(args)
        elif args.command == 'test-tokens':
            return cli.cmd_test_tokens(args)
        elif args.command == 'test-pcr':
            return cli.cmd_test_pcr(args)
        elif args.command == 'start-emulation':
            return cli.cmd_start_emulation(args)
        elif args.command == 'run-tests':
            return cli.cmd_run_tests(args)
        elif args.command == 'analyze-npu':
            return cli.cmd_analyze_npu(args)
        elif args.command == 'bridge-test':
            return cli.cmd_bridge_test(args)
        elif args.command == 'info':
            return cli.cmd_info(args)
        else:
            parser.print_help()
            return 1

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        cli.cleanup()
        return 130
    except Exception as e:
        logger.error(f"Command execution error: {e}")
        return 1
    finally:
        cli.cleanup()


if __name__ == "__main__":
    sys.exit(main())