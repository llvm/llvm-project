#!/usr/bin/env python3
"""
SMBIOS Token to DSMIL Device Correlation Analysis Script
========================================================

Analyzes the correlation between 72 SMBIOS tokens (0x0480-0x04C7) and 72 DSMIL devices.
Maps tokens to potential DSMIL functions based on architectural patterns.

Author: DSMIL Analysis Framework
Date: 2025-09-01
Hardware: Dell Latitude 5450 MIL-SPEC
"""

import json
import sys
import os
import subprocess
import time
import argparse
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# Configuration
TOKEN_START = 0x0480
TOKEN_END = 0x04C7
TOTAL_TOKENS = 72
DEVICES_PER_GROUP = 12
TOTAL_GROUPS = 6

class DSMILFunction(Enum):
    """Potential DSMIL device functions based on Dell architecture patterns"""
    POWER_MANAGEMENT = "power_management"
    THERMAL_CONTROL = "thermal_control"
    SECURITY_MODULE = "security_module"
    MEMORY_CONTROL = "memory_control"
    IO_CONTROLLER = "io_controller"
    NETWORK_INTERFACE = "network_interface"
    STORAGE_CONTROL = "storage_control"
    DISPLAY_CONTROL = "display_control"
    AUDIO_CONTROL = "audio_control"
    SENSOR_HUB = "sensor_hub"
    ACCELEROMETER = "accelerometer"
    UNKNOWN = "unknown"

@dataclass
class TokenInfo:
    """Information about a single SMBIOS token"""
    token_id: int
    hex_id: str
    group_id: int
    device_id: int
    sequential_index: int
    potential_function: DSMILFunction
    confidence: float
    accessible: bool = False
    current_value: Optional[int] = None
    description: str = ""

@dataclass
class DSMILGroup:
    """DSMIL device group information"""
    group_id: int
    start_token: int
    end_token: int
    devices: List[TokenInfo]
    primary_function: DSMILFunction
    group_description: str

@dataclass
class CorrelationAnalysis:
    """Complete correlation analysis results"""
    total_tokens: int
    analyzed_tokens: int
    groups: List[DSMILGroup]
    system_info: Dict[str, Any]
    analysis_timestamp: str
    confidence_metrics: Dict[str, float]

class DSMILTokenAnalyzer:
    """Main analyzer for SMBIOS token to DSMIL device correlation"""
    
    def __init__(self, dry_run: bool = True, verbose: bool = False):
        self.dry_run = dry_run
        self.verbose = verbose
        self.logger = self._setup_logging()
        self.system_info = {}
        self.thermal_baseline = None
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.DEBUG if self.verbose else logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def check_system_safety(self) -> bool:
        """Check system safety before token analysis"""
        self.logger.info("Performing system safety checks...")
        
        # Check thermal status
        try:
            thermal_info = self._get_thermal_info()
            if thermal_info.get('max_temp', 0) > 85:
                self.logger.warning(f"High system temperature: {thermal_info['max_temp']}°C")
                return False
        except Exception as e:
            self.logger.warning(f"Could not read thermal info: {e}")
        
        # Check if running as root (required for SMBIOS access)
        if os.geteuid() != 0 and not self.dry_run:
            self.logger.error("Root privileges required for SMBIOS token access")
            return False
        
        # Check Dell system
        dmi_info = self._get_dmi_info()
        if 'Dell' not in dmi_info.get('manufacturer', ''):
            self.logger.warning("Non-Dell system detected - token mapping may not apply")
        
        self.logger.info("System safety checks passed")
        return True
    
    def _get_thermal_info(self) -> Dict[str, Any]:
        """Get current thermal information"""
        thermal_info = {'temps': [], 'max_temp': 0}
        
        try:
            # Read from thermal zones
            thermal_base = '/sys/class/thermal'
            if os.path.exists(thermal_base):
                for zone in os.listdir(thermal_base):
                    if zone.startswith('thermal_zone'):
                        temp_file = f"{thermal_base}/{zone}/temp"
                        if os.path.exists(temp_file):
                            with open(temp_file, 'r') as f:
                                temp_millidegrees = int(f.read().strip())
                                temp_celsius = temp_millidegrees / 1000
                                thermal_info['temps'].append(temp_celsius)
                                thermal_info['max_temp'] = max(thermal_info['max_temp'], temp_celsius)
        except Exception as e:
            self.logger.debug(f"Error reading thermal zones: {e}")
        
        return thermal_info
    
    def _get_dmi_info(self) -> Dict[str, str]:
        """Get DMI/SMBIOS system information"""
        dmi_info = {}
        dmi_fields = {
            'manufacturer': '/sys/class/dmi/id/sys_vendor',
            'product_name': '/sys/class/dmi/id/product_name',
            'product_version': '/sys/class/dmi/id/product_version',
            'bios_version': '/sys/class/dmi/id/bios_version',
            'bios_date': '/sys/class/dmi/id/bios_date'
        }
        
        for field, path in dmi_fields.items():
            try:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        dmi_info[field] = f.read().strip()
            except Exception as e:
                self.logger.debug(f"Could not read {field}: {e}")
        
        return dmi_info
    
    def analyze_token_accessibility(self) -> Dict[int, bool]:
        """Analyze which tokens are accessible via dell-smbios-wmi"""
        self.logger.info("Analyzing token accessibility...")
        accessible_tokens = {}
        
        if self.dry_run:
            # In dry-run mode, simulate some accessible tokens
            for token_id in range(TOKEN_START, TOKEN_END + 1):
                # Simulate ~70% accessibility with some pattern
                accessible_tokens[token_id] = (token_id % 3 != 0)
        else:
            # Real accessibility check using dell-smbios-wmi
            for token_id in range(TOKEN_START, TOKEN_END + 1):
                try:
                    # Attempt to read token without modification
                    result = subprocess.run([
                        'python3', '-c', f'''
import subprocess
try:
    result = subprocess.run(["dcli", "BootOrder", "GetTokenValue", "{hex(token_id)}"], 
                          capture_output=True, text=True, timeout=5)
    exit(0 if result.returncode == 0 else 1)
except:
    exit(1)
'''
                    ], capture_output=True, timeout=10)
                    accessible_tokens[token_id] = result.returncode == 0
                except Exception as e:
                    self.logger.debug(f"Token {hex(token_id)} accessibility check failed: {e}")
                    accessible_tokens[token_id] = False
        
        accessible_count = sum(accessible_tokens.values())
        self.logger.info(f"Found {accessible_count}/{TOTAL_TOKENS} accessible tokens")
        return accessible_tokens
    
    def generate_function_hypothesis(self, group_id: int, device_id: int, token_id: int) -> Tuple[DSMILFunction, float]:
        """Generate hypothesis for token function based on patterns"""
        
        # Dell architectural patterns for DSMIL devices
        function_patterns = {
            0: {  # Group 0: Core system functions
                0: (DSMILFunction.POWER_MANAGEMENT, 0.9),
                1: (DSMILFunction.THERMAL_CONTROL, 0.9),
                2: (DSMILFunction.SECURITY_MODULE, 0.8),
                3: (DSMILFunction.MEMORY_CONTROL, 0.8),
                4: (DSMILFunction.IO_CONTROLLER, 0.7),
                5: (DSMILFunction.NETWORK_INTERFACE, 0.7),
                6: (DSMILFunction.STORAGE_CONTROL, 0.7),
                7: (DSMILFunction.DISPLAY_CONTROL, 0.6),
                8: (DSMILFunction.AUDIO_CONTROL, 0.6),
                9: (DSMILFunction.SENSOR_HUB, 0.6),
                10: (DSMILFunction.ACCELEROMETER, 0.5),
                11: (DSMILFunction.UNKNOWN, 0.3)
            },
            1: {  # Group 1: Secondary system functions
                0: (DSMILFunction.THERMAL_CONTROL, 0.8),
                1: (DSMILFunction.POWER_MANAGEMENT, 0.8),
                2: (DSMILFunction.MEMORY_CONTROL, 0.7),
                3: (DSMILFunction.IO_CONTROLLER, 0.7),
                4: (DSMILFunction.NETWORK_INTERFACE, 0.6),
                5: (DSMILFunction.STORAGE_CONTROL, 0.6),
                6: (DSMILFunction.SECURITY_MODULE, 0.6),
                7: (DSMILFunction.DISPLAY_CONTROL, 0.5),
                8: (DSMILFunction.AUDIO_CONTROL, 0.5),
                9: (DSMILFunction.SENSOR_HUB, 0.5),
                10: (DSMILFunction.ACCELEROMETER, 0.4),
                11: (DSMILFunction.UNKNOWN, 0.3)
            }
        }
        
        # Pattern for groups 2-5 (decreasing confidence)
        if group_id not in function_patterns:
            base_confidence = max(0.7 - (group_id - 2) * 0.1, 0.3)
            device_functions = [
                DSMILFunction.POWER_MANAGEMENT,
                DSMILFunction.THERMAL_CONTROL,
                DSMILFunction.MEMORY_CONTROL,
                DSMILFunction.IO_CONTROLLER,
                DSMILFunction.NETWORK_INTERFACE,
                DSMILFunction.STORAGE_CONTROL,
                DSMILFunction.SECURITY_MODULE,
                DSMILFunction.DISPLAY_CONTROL,
                DSMILFunction.AUDIO_CONTROL,
                DSMILFunction.SENSOR_HUB,
                DSMILFunction.ACCELEROMETER,
                DSMILFunction.UNKNOWN
            ]
            
            if device_id < len(device_functions):
                return device_functions[device_id], base_confidence
            else:
                return DSMILFunction.UNKNOWN, 0.2
        
        return function_patterns[group_id].get(device_id, (DSMILFunction.UNKNOWN, 0.2))
    
    def analyze_tokens(self) -> CorrelationAnalysis:
        """Perform complete token correlation analysis"""
        self.logger.info("Starting SMBIOS token to DSMIL device correlation analysis...")
        
        # Get system information
        self.system_info = self._get_dmi_info()
        self.system_info.update(self._get_thermal_info())
        
        # Analyze token accessibility
        accessible_tokens = self.analyze_token_accessibility()
        
        # Generate token information
        all_tokens = []
        groups = []
        
        for group_id in range(TOTAL_GROUPS):
            group_tokens = []
            
            for device_id in range(DEVICES_PER_GROUP):
                sequential_index = group_id * DEVICES_PER_GROUP + device_id
                token_id = TOKEN_START + sequential_index
                
                function, confidence = self.generate_function_hypothesis(group_id, device_id, token_id)
                
                token_info = TokenInfo(
                    token_id=token_id,
                    hex_id=hex(token_id),
                    group_id=group_id,
                    device_id=device_id,
                    sequential_index=sequential_index,
                    potential_function=function,
                    confidence=confidence,
                    accessible=accessible_tokens.get(token_id, False),
                    description=f"Group {group_id}, Device {device_id}: {function.value}"
                )
                
                group_tokens.append(token_info)
                all_tokens.append(token_info)
            
            # Determine primary group function
            primary_function = max(group_tokens, key=lambda t: t.confidence).potential_function
            
            group = DSMILGroup(
                group_id=group_id,
                start_token=TOKEN_START + group_id * DEVICES_PER_GROUP,
                end_token=TOKEN_START + (group_id + 1) * DEVICES_PER_GROUP - 1,
                devices=group_tokens,
                primary_function=primary_function,
                group_description=f"DSMIL Group {group_id}: {primary_function.value} cluster"
            )
            
            groups.append(group)
        
        # Calculate confidence metrics
        confidence_metrics = {
            'average_confidence': sum(t.confidence for t in all_tokens) / len(all_tokens),
            'accessibility_ratio': sum(1 for t in all_tokens if t.accessible) / len(all_tokens),
            'high_confidence_count': sum(1 for t in all_tokens if t.confidence > 0.7),
            'identified_functions': len(set(t.potential_function for t in all_tokens if t.potential_function != DSMILFunction.UNKNOWN))
        }
        
        analysis = CorrelationAnalysis(
            total_tokens=TOTAL_TOKENS,
            analyzed_tokens=len(all_tokens),
            groups=groups,
            system_info=self.system_info,
            analysis_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            confidence_metrics=confidence_metrics
        )
        
        self.logger.info("Token correlation analysis complete")
        return analysis
    
    def save_json_report(self, analysis: CorrelationAnalysis, output_file: str) -> None:
        """Save analysis results to JSON file"""
        
        def convert_enum(obj):
            if isinstance(obj, DSMILFunction):
                return obj.value
            elif hasattr(obj, '__dict__'):
                return {k: convert_enum(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, list):
                return [convert_enum(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert_enum(v) for k, v in obj.items()}
            return obj
        
        json_data = convert_enum(asdict(analysis))
        
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        self.logger.info(f"JSON report saved to {output_file}")
    
    def generate_human_report(self, analysis: CorrelationAnalysis) -> str:
        """Generate human-readable analysis report"""
        
        report = []
        report.append("=" * 80)
        report.append("SMBIOS TOKEN TO DSMIL DEVICE CORRELATION ANALYSIS")
        report.append("=" * 80)
        report.append(f"Analysis Date: {analysis.analysis_timestamp}")
        report.append(f"System: {analysis.system_info.get('manufacturer', 'Unknown')} {analysis.system_info.get('product_name', 'Unknown')}")
        report.append(f"BIOS: {analysis.system_info.get('bios_version', 'Unknown')} ({analysis.system_info.get('bios_date', 'Unknown')})")
        report.append(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE ANALYSIS'}")
        report.append("")
        
        # Summary statistics
        report.append("ANALYSIS SUMMARY")
        report.append("-" * 40)
        report.append(f"Total Tokens Analyzed: {analysis.total_tokens}")
        report.append(f"Token Range: {hex(TOKEN_START)}-{hex(TOKEN_END)}")
        report.append(f"Average Confidence: {analysis.confidence_metrics['average_confidence']:.2f}")
        report.append(f"Accessibility Ratio: {analysis.confidence_metrics['accessibility_ratio']:.2%}")
        report.append(f"High Confidence Tokens: {analysis.confidence_metrics['high_confidence_count']}")
        report.append(f"Identified Functions: {analysis.confidence_metrics['identified_functions']}")
        
        if 'max_temp' in analysis.system_info:
            report.append(f"Current Max Temperature: {analysis.system_info['max_temp']:.1f}°C")
        
        report.append("")
        
        # Group analysis
        report.append("DSMIL GROUP ANALYSIS")
        report.append("-" * 40)
        
        for group in analysis.groups:
            report.append(f"\nGroup {group.group_id}: {group.group_description}")
            report.append(f"  Token Range: {hex(group.start_token)}-{hex(group.end_token)}")
            report.append(f"  Primary Function: {group.primary_function.value}")
            report.append(f"  Devices: {len(group.devices)}")
            
            accessible_count = sum(1 for d in group.devices if d.accessible)
            report.append(f"  Accessible Tokens: {accessible_count}/{len(group.devices)}")
            
            # Top confidence devices
            top_devices = sorted(group.devices, key=lambda d: d.confidence, reverse=True)[:3]
            report.append("  Top Confidence Devices:")
            for device in top_devices:
                report.append(f"    {device.hex_id}: {device.potential_function.value} ({device.confidence:.2f})")
        
        report.append("")
        
        # Detailed token mapping
        report.append("DETAILED TOKEN MAPPING")
        report.append("-" * 40)
        
        for group in analysis.groups:
            report.append(f"\n--- Group {group.group_id} Details ---")
            for device in group.devices:
                status = "✓" if device.accessible else "✗"
                report.append(f"{device.hex_id} [{status}] G{device.group_id}D{device.device_id:02d}: {device.potential_function.value:20} (conf: {device.confidence:.2f})")
        
        report.append("")
        
        # Safety recommendations
        report.append("SAFETY RECOMMENDATIONS")
        report.append("-" * 40)
        report.append("1. Always run thermal monitoring during token activation")
        report.append("2. Test tokens individually, never in batch")
        report.append("3. Focus on high-confidence, accessible tokens first")
        report.append("4. Monitor system stability after each token change")
        report.append("5. Have thermal throttling and emergency shutdown ready")
        report.append("6. Back up current BIOS settings before modification")
        
        report.append("")
        
        # Next steps
        report.append("RECOMMENDED NEXT STEPS")
        report.append("-" * 40)
        
        high_confidence_accessible = [
            token for group in analysis.groups 
            for token in group.devices 
            if token.confidence > 0.7 and token.accessible
        ]
        
        if high_confidence_accessible:
            report.append("High-priority tokens for testing:")
            for token in sorted(high_confidence_accessible, key=lambda t: t.confidence, reverse=True)[:10]:
                report.append(f"  {token.hex_id}: {token.potential_function.value} (confidence: {token.confidence:.2f})")
        
        report.append("\n1. Start with Group 0 power management tokens")
        report.append("2. Validate thermal control tokens in controlled environment")
        report.append("3. Test security module tokens with caution")
        report.append("4. Map memory control tokens to actual memory regions")
        report.append("5. Correlate findings with actual DSMIL device behavior")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_human_report(self, analysis: CorrelationAnalysis, output_file: str) -> None:
        """Save human-readable report to file"""
        report = self.generate_human_report(analysis)
        
        with open(output_file, 'w') as f:
            f.write(report)
        
        self.logger.info(f"Human-readable report saved to {output_file}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="SMBIOS Token to DSMIL Device Correlation Analysis")
    parser.add_argument("--live", action="store_true", help="Perform live analysis (requires root)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument("--json", "-j", help="Output JSON file", default="dsmil_token_correlation.json")
    parser.add_argument("--report", "-r", help="Output report file", default="dsmil_token_analysis_report.txt")
    parser.add_argument("--no-safety", action="store_true", help="Skip safety checks")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = DSMILTokenAnalyzer(dry_run=not args.live, verbose=args.verbose)
    
    try:
        # Safety checks
        if not args.no_safety and not analyzer.check_system_safety():
            print("System safety checks failed. Use --no-safety to override.", file=sys.stderr)
            return 1
        
        # Perform analysis
        analysis = analyzer.analyze_tokens()
        
        # Generate outputs
        analyzer.save_json_report(analysis, args.json)
        analyzer.save_human_report(analysis, args.report)
        
        # Print summary to console
        print("\n" + "=" * 60)
        print("DSMIL TOKEN CORRELATION ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"Mode: {'LIVE ANALYSIS' if args.live else 'DRY RUN'}")
        print(f"Tokens Analyzed: {analysis.total_tokens}")
        print(f"Average Confidence: {analysis.confidence_metrics['average_confidence']:.2f}")
        print(f"Accessible Tokens: {analysis.confidence_metrics['accessibility_ratio']:.1%}")
        print(f"JSON Output: {args.json}")
        print(f"Report Output: {args.report}")
        
        if not args.live:
            print("\nNote: This was a dry-run analysis. Use --live for actual token testing.")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Analysis failed: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())