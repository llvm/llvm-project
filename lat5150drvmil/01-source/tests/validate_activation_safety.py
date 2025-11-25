#!/usr/bin/env python3
"""
⚠️ DEPRECATED - Use dsmil_device_activation.py built-in safety checks ⚠️

This script is deprecated. Safety validation is now integrated into the
comprehensive device activation framework with automatic pre-activation checks.

For device activation with built-in safety:
    python3 02-ai-engine/dsmil_device_activation.py --device <ID>

Date Deprecated: 2025-11-07
Superseded by: 02-ai-engine/dsmil_device_activation.py (includes safety validation)

OLD DESCRIPTION:
Military Token Activation Safety Validator
Pre-activation safety verification and system analysis

MISSION: Comprehensive safety validation before military token activation
- Verifies quarantine list integrity
- Analyzes thermal baselines
- Checks DSMIL module status
- Validates Dell WMI interface accessibility
- Performs dry-run simulation
- Generates safety assessment report
"""

import os
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

class ActivationSafetyValidator:
    """Comprehensive safety validation for military token activation"""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Critical safety configuration
        self.quarantined_tokens = {
            0x8009, 0x800A, 0x800B, 0x8019, 0x8029
        }
        
        self.target_tokens = {
            0x8000: "Primary Command Interface",
            0x8014: "Secure Communications", 
            0x801E: "Tactical Display Control",
            0x8028: "Power Management Unit",
            0x8032: "Memory Protection",
            0x803C: "I/O Security Controller",
            0x8046: "Network Security Module",
            0x8050: "Storage Encryption",
            0x805A: "Sensor Array",
            0x8064: "Auxiliary Systems"
        }
        
        self.safety_report = {
            'timestamp': datetime.now().isoformat(),
            'validation_status': 'PENDING',
            'safety_checks': {},
            'thermal_analysis': {},
            'system_readiness': {},
            'recommendations': [],
            'risks_identified': []
        }
    
    def _log(self, message: str, level: str = "INFO"):
        """Safety validation logging"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [SAFETY-{level}] {message}")
    
    def _get_max_temperature(self) -> float:
        """Get current maximum system temperature"""
        max_temp = 0.0
        try:
            for zone_path in Path('/sys/class/thermal').glob('thermal_zone*/temp'):
                temp_raw = int(zone_path.read_text().strip())
                temp_celsius = temp_raw / 1000.0
                max_temp = max(max_temp, temp_celsius)
        except Exception as e:
            self._log(f"Temperature reading error: {e}", "WARN")
        return max_temp
    
    def validate_quarantine_integrity(self) -> bool:
        """Verify quarantine list is properly configured"""
        self._log("Validating quarantine list integrity...")
        
        quarantine_check = {
            'total_quarantined': len(self.quarantined_tokens),
            'quarantined_list': [f"0x{token:04x}" for token in self.quarantined_tokens],
            'conflicts_detected': [],
            'integrity_status': 'PASS'
        }
        
        # Check for conflicts between target and quarantine lists
        conflicts = set(self.target_tokens.keys()) & self.quarantined_tokens
        if conflicts:
            quarantine_check['conflicts_detected'] = [f"0x{token:04x}" for token in conflicts]
            quarantine_check['integrity_status'] = 'FAIL'
            self._log(f"CRITICAL: {len(conflicts)} tokens in both target and quarantine lists!", "ERROR")
            for token in conflicts:
                self.safety_report['risks_identified'].append(f"Token 0x{token:04x} in both lists")
        else:
            self._log(f"✅ No conflicts: {len(self.quarantined_tokens)} tokens safely quarantined")
        
        self.safety_report['safety_checks']['quarantine_integrity'] = quarantine_check
        return quarantine_check['integrity_status'] == 'PASS'
    
    def validate_thermal_conditions(self) -> bool:
        """Analyze thermal conditions and safety margins"""
        self._log("Analyzing thermal conditions...")
        
        current_temp = self._get_max_temperature()
        thermal_analysis = {
            'current_temperature': current_temp,
            'safety_thresholds': {
                'warning': 90.0,
                'critical': 95.0,
                'emergency': 100.0
            },
            'safety_margins': {},
            'thermal_status': 'UNKNOWN'
        }
        
        # Calculate safety margins
        thermal_analysis['safety_margins'] = {
            'to_warning': 90.0 - current_temp,
            'to_critical': 95.0 - current_temp,
            'to_emergency': 100.0 - current_temp
        }
        
        # Determine thermal status
        if current_temp >= 95.0:
            thermal_analysis['thermal_status'] = 'CRITICAL'
            self._log(f"🚨 CRITICAL: Temperature {current_temp:.1f}°C exceeds safe limits", "ERROR")
            self.safety_report['risks_identified'].append(f"Critical temperature: {current_temp:.1f}°C")
        elif current_temp >= 90.0:
            thermal_analysis['thermal_status'] = 'WARNING'
            self._log(f"⚠️  WARNING: Temperature {current_temp:.1f}°C approaching limits", "WARN")
            self.safety_report['recommendations'].append("Consider thermal cooldown before activation")
        else:
            thermal_analysis['thermal_status'] = 'SAFE'
            self._log(f"✅ Thermal conditions safe: {current_temp:.1f}°C")
        
        self.safety_report['thermal_analysis'] = thermal_analysis
        return thermal_analysis['thermal_status'] in ['SAFE', 'WARNING']
    
    def validate_system_prerequisites(self) -> bool:
        """Check system prerequisites for safe activation"""
        self._log("Validating system prerequisites...")
        
        prerequisites = {
            'dsmil_module_loaded': False,
            'dell_wmi_available': False,
            'smbios_tools_available': False,
            'root_privileges': False,
            'system_stability': 'UNKNOWN'
        }
        
        # Check DSMIL module
        try:
            result = subprocess.run(['lsmod'], capture_output=True, text=True)
            prerequisites['dsmil_module_loaded'] = 'dsmil_72dev' in result.stdout
            if prerequisites['dsmil_module_loaded']:
                self._log("✅ DSMIL kernel module loaded")
            else:
                self._log("❌ DSMIL kernel module NOT loaded", "ERROR")
                self.safety_report['risks_identified'].append("DSMIL module not loaded")
        except Exception as e:
            self._log(f"Error checking DSMIL module: {e}", "ERROR")
        
        # Check Dell WMI interface
        dell_wmi_path = Path("/sys/devices/virtual/firmware-attributes/dell-wmi-sysman")
        prerequisites['dell_wmi_available'] = dell_wmi_path.exists()
        if prerequisites['dell_wmi_available']:
            self._log("✅ Dell WMI interface available")
        else:
            self._log("❌ Dell WMI interface NOT available", "ERROR")
            self.safety_report['risks_identified'].append("Dell WMI interface missing")
        
        # Check smbios-token-ctl availability
        try:
            result = subprocess.run(['which', 'smbios-token-ctl'], capture_output=True)
            prerequisites['smbios_tools_available'] = result.returncode == 0
            if prerequisites['smbios_tools_available']:
                self._log("✅ smbios-token-ctl available")
            else:
                self._log("❌ smbios-token-ctl NOT available", "ERROR")
                self.safety_report['risks_identified'].append("smbios-token-ctl not installed")
        except Exception as e:
            self._log(f"Error checking smbios tools: {e}", "ERROR")
        
        # Check root privileges
        prerequisites['root_privileges'] = os.geteuid() == 0
        if prerequisites['root_privileges']:
            self._log("✅ Root privileges available")
        else:
            self._log("⚠️  Root privileges required for activation", "WARN")
            self.safety_report['recommendations'].append("Run with sudo for full validation")
        
        self.safety_report['system_readiness'] = prerequisites
        
        # Calculate readiness score
        ready_count = sum([
            prerequisites['dsmil_module_loaded'],
            prerequisites['dell_wmi_available'],
            prerequisites['smbios_tools_available']
        ])
        
        return ready_count >= 2  # At least 2 of 3 critical prerequisites
    
    def perform_dry_run_simulation(self) -> bool:
        """Simulate activation sequence without making changes"""
        self._log("Performing dry-run simulation...")
        
        simulation_results = {
            'total_tokens': len(self.target_tokens),
            'accessible_tokens': 0,
            'blocked_tokens': 0,
            'unknown_tokens': 0,
            'simulation_success': False
        }
        
        for token_id, description in self.target_tokens.items():
            self._log(f"  Simulating 0x{token_id:04x}: {description}")
            
            # Check if token would be blocked by quarantine
            if token_id in self.quarantined_tokens:
                self._log(f"    ❌ BLOCKED: Quarantined", "ERROR")
                simulation_results['blocked_tokens'] += 1
                continue
            
            # Attempt to read token (safe operation)
            try:
                result = subprocess.run([
                    'sudo', '-S', 'smbios-token-ctl',
                    f'--token-id=0x{token_id:04x}', '--get'
                ], input="1786\n", text=True, capture_output=True, timeout=10)
                
                if result.returncode == 0:
                    self._log(f"    ✅ Accessible: {result.stdout.strip()}")
                    simulation_results['accessible_tokens'] += 1
                else:
                    self._log(f"    ⚠️  Not accessible: {result.stderr.strip()}")
                    simulation_results['unknown_tokens'] += 1
                    
            except subprocess.TimeoutExpired:
                self._log(f"    ⚠️  Timeout reading token", "WARN")
                simulation_results['unknown_tokens'] += 1
            except Exception as e:
                self._log(f"    ❌ Error: {e}", "ERROR")
                simulation_results['unknown_tokens'] += 1
        
        # Determine simulation success
        simulation_results['simulation_success'] = (
            simulation_results['accessible_tokens'] > 0 and
            simulation_results['blocked_tokens'] == 0
        )
        
        self._log(f"Simulation results:")
        self._log(f"  Accessible: {simulation_results['accessible_tokens']}/{simulation_results['total_tokens']}")
        self._log(f"  Blocked: {simulation_results['blocked_tokens']}")
        self._log(f"  Unknown: {simulation_results['unknown_tokens']}")
        
        self.safety_report['safety_checks']['dry_run'] = simulation_results
        return simulation_results['simulation_success']
    
    def generate_safety_assessment(self) -> str:
        """Generate comprehensive safety assessment"""
        self._log("Generating comprehensive safety assessment...")
        
        # Run all validation checks
        quarantine_ok = self.validate_quarantine_integrity()
        thermal_ok = self.validate_thermal_conditions()  
        prerequisites_ok = self.validate_system_prerequisites()
        simulation_ok = self.perform_dry_run_simulation()
        
        # Calculate overall safety score
        safety_checks_passed = sum([quarantine_ok, thermal_ok, prerequisites_ok, simulation_ok])
        safety_score = safety_checks_passed / 4.0 * 100
        
        # Determine validation status
        if safety_score >= 75 and quarantine_ok:
            self.safety_report['validation_status'] = 'SAFE_TO_PROCEED'
            status_color = '🟢'
        elif safety_score >= 50:
            self.safety_report['validation_status'] = 'PROCEED_WITH_CAUTION'
            status_color = '🟡'
        else:
            self.safety_report['validation_status'] = 'DO_NOT_PROCEED'
            status_color = '🔴'
        
        # Add final recommendations
        if thermal_ok and prerequisites_ok:
            self.safety_report['recommendations'].append("Monitor thermal conditions during activation")
        if not prerequisites_ok:
            self.safety_report['recommendations'].append("Resolve system prerequisites before activation")
        if len(self.safety_report['risks_identified']) > 0:
            self.safety_report['recommendations'].append("Address all identified risks before proceeding")
        
        # Generate report
        self._log("="*80)
        self._log("MILITARY TOKEN ACTIVATION SAFETY ASSESSMENT")
        self._log("="*80)
        self._log(f"Overall Status: {status_color} {self.safety_report['validation_status']}")
        self._log(f"Safety Score: {safety_score:.1f}%")
        self._log(f"")
        
        self._log(f"Safety Checks:")
        self._log(f"  Quarantine Integrity: {'✅ PASS' if quarantine_ok else '❌ FAIL'}")
        self._log(f"  Thermal Conditions: {'✅ SAFE' if thermal_ok else '❌ UNSAFE'}")
        self._log(f"  System Prerequisites: {'✅ READY' if prerequisites_ok else '❌ NOT READY'}")
        self._log(f"  Dry-Run Simulation: {'✅ SUCCESS' if simulation_ok else '❌ FAILED'}")
        
        if self.safety_report['risks_identified']:
            self._log(f"")
            self._log(f"⚠️  RISKS IDENTIFIED:")
            for risk in self.safety_report['risks_identified']:
                self._log(f"    • {risk}")
        
        if self.safety_report['recommendations']:
            self._log(f"")
            self._log(f"💡 RECOMMENDATIONS:")
            for rec in self.safety_report['recommendations']:
                self._log(f"    • {rec}")
        
        # Save detailed report
        report_file = Path(f"safety_assessment_{self.timestamp}.json")
        with open(report_file, 'w') as f:
            json.dump(self.safety_report, f, indent=2)
        
        self._log(f"")
        self._log(f"📋 Detailed assessment saved: {report_file}")
        self._log("="*80)
        
        return self.safety_report['validation_status']

def main():
    """Main safety validation execution"""
    print("Military Token Activation Safety Validator")
    print("Dell Latitude 5450 MIL-SPEC - HARDWARE-DELL & SECURITY Agents")
    print("="*60)
    
    try:
        validator = ActivationSafetyValidator()
        status = validator.generate_safety_assessment()
        
        # Exit codes for automation
        if status == 'SAFE_TO_PROCEED':
            sys.exit(0)
        elif status == 'PROCEED_WITH_CAUTION':
            sys.exit(1)  
        else:
            sys.exit(2)  # DO_NOT_PROCEED
            
    except KeyboardInterrupt:
        print("\n🛑 Safety validation interrupted")
        sys.exit(130)
    except Exception as e:
        print(f"💥 Error during safety validation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import sys
    main()