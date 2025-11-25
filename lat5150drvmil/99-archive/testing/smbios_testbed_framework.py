#!/usr/bin/env python3
"""
Dell Latitude 5450 MIL-SPEC SMBIOS Token Testing Framework
==========================================================

TESTBED Agent - Systematic SMBIOS token testing with comprehensive safety mechanisms

Target Hardware: Dell Latitude 5450 MIL-SPEC (JRTC1 Training Variant)
Thermal Profile: 100°C normal operation (warning at 95°C, critical at 100°C)
Target Tokens: 0x0480-0x04C7 (72 tokens in 6 groups of 12)

Safety Features:
- Real-time thermal monitoring with emergency stops
- Incremental testing with immediate rollback capability
- DSMIL response correlation
- Automated safety validation
- Ubuntu 24.04 and Debian Trixie compatibility

Author: TESTBED Agent
Version: 1.0.0
Date: 2025-09-01
"""

import os
import sys
import time
import json
import subprocess
import threading
import tempfile
import shutil
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path

# System monitoring imports
try:
    import psutil
except ImportError:
    print("Installing psutil...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--user", "psutil"], check=True)
    import psutil

@dataclass
class TokenTestResult:
    """Results from a single token test"""
    token: str
    test_time: datetime
    initial_value: Optional[str]
    activation_successful: bool
    final_value: Optional[str] 
    dsmil_response: List[str]
    thermal_readings: List[float]
    rollback_successful: bool
    errors: List[str]
    warnings: List[str]
    system_impact: Dict[str, Any]

@dataclass
class TestSession:
    """Complete testing session information"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    token_range: Tuple[str, str]
    total_tokens_tested: int
    successful_tests: int
    failed_tests: int
    emergency_stops: int
    results: List[TokenTestResult]

class ThermalMonitor:
    """Real-time thermal monitoring with emergency stop capability"""
    
    def __init__(self, warning_threshold: float = 95.0, critical_threshold: float = 100.0, 
                 emergency_threshold: float = 105.0):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.emergency_threshold = emergency_threshold
        self.monitoring = False
        self.emergency_triggered = False
        self.monitor_thread = None
        self.thermal_data = []
        
    def start_monitoring(self, callback=None):
        """Start continuous thermal monitoring"""
        self.monitoring = True
        self.emergency_triggered = False
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(callback,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop thermal monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
            
    def _monitor_loop(self, callback):
        """Continuous thermal monitoring loop"""
        while self.monitoring:
            try:
                temperatures = self._get_thermal_readings()
                if temperatures:
                    max_temp = max(temperatures)
                    self.thermal_data.append({
                        'timestamp': datetime.now(timezone.utc),
                        'temperatures': temperatures,
                        'max_temp': max_temp
                    })
                    
                    # Check thresholds
                    if max_temp >= self.emergency_threshold:
                        self.emergency_triggered = True
                        self.monitoring = False
                        if callback:
                            callback("EMERGENCY", max_temp, temperatures)
                        break
                    elif max_temp >= self.critical_threshold:
                        if callback:
                            callback("CRITICAL", max_temp, temperatures)
                    elif max_temp >= self.warning_threshold:
                        if callback:
                            callback("WARNING", max_temp, temperatures)
                            
                time.sleep(1)  # Check every second
                
            except Exception as e:
                print(f"Thermal monitoring error: {e}")
                time.sleep(5)  # Retry after error
                
    def _get_thermal_readings(self) -> List[float]:
        """Get current thermal readings from all sensors"""
        temperatures = []
        try:
            # psutil thermal sensors
            if hasattr(psutil, 'sensors_temperatures'):
                temps = psutil.sensors_temperatures()
                for name, entries in temps.items():
                    for entry in entries:
                        if entry.current:
                            temperatures.append(entry.current)
            
            # Additional thermal zone reading
            thermal_zones = Path("/sys/class/thermal").glob("thermal_zone*")
            for zone in thermal_zones:
                temp_file = zone / "temp"
                if temp_file.exists():
                    try:
                        temp_raw = int(temp_file.read_text().strip())
                        temp_celsius = temp_raw / 1000.0
                        temperatures.append(temp_celsius)
                    except (ValueError, OSError):
                        continue
                        
        except Exception as e:
            print(f"Error reading thermal sensors: {e}")
            
        return temperatures
        
    def get_current_max_temp(self) -> float:
        """Get current maximum temperature"""
        temps = self._get_thermal_readings()
        return max(temps) if temps else 0.0

class SMBIOSTokenTester:
    """Core SMBIOS token testing framework with safety mechanisms"""
    
    def __init__(self, work_dir: str = "/home/john/LAT5150DRVMIL"):
        self.work_dir = Path(work_dir)
        self.testing_dir = self.work_dir / "testing"
        self.testing_dir.mkdir(exist_ok=True)
        
        # Initialize thermal monitoring
        self.thermal_monitor = ThermalMonitor()
        
        # Initialize safety systems
        self.safety_active = True
        self.emergency_stop_triggered = False
        
        # Test tracking
        self.current_session: Optional[TestSession] = None
        self.test_results: List[TokenTestResult] = []
        
        # Target token ranges
        self.TARGET_RANGES = {
            'Range_0480': list(range(0x0480, 0x04C8)),  # 72 tokens: 0x0480-0x04C7
            'Range_0400': list(range(0x0400, 0x0448)),  # Alternative range
            'Range_0500': list(range(0x0500, 0x0548))   # Alternative range
        }
        
        self.DSMIL_GROUPS = {
            'Group_0': list(range(0x0480, 0x048C)),  # Tokens 0x0480-0x048B (12 tokens)
            'Group_1': list(range(0x048C, 0x0498)),  # Tokens 0x048C-0x0497 (12 tokens)
            'Group_2': list(range(0x0498, 0x04A4)),  # Tokens 0x0498-0x04A3 (12 tokens)
            'Group_3': list(range(0x04A4, 0x04B0)),  # Tokens 0x04A4-0x04AF (12 tokens)
            'Group_4': list(range(0x04B0, 0x04BC)),  # Tokens 0x04B0-0x04BB (12 tokens)
            'Group_5': list(range(0x04BC, 0x04C8))   # Tokens 0x04BC-0x04C7 (12 tokens)
        }
        
    def create_test_session(self, token_range: str = "Range_0480") -> TestSession:
        """Create a new testing session"""
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Support special single_token session type
        if token_range == "single_token":
            tokens = [0x0480]  # Test first token
            start_token = f"0x0480"
            end_token = f"0x0480"
        elif token_range not in self.TARGET_RANGES:
            raise ValueError(f"Unknown token range: {token_range}")
        else:
            tokens = self.TARGET_RANGES[token_range]
            start_token = f"0x{tokens[0]:04X}"
            end_token = f"0x{tokens[-1]:04X}"
        
        session = TestSession(
            session_id=session_id,
            start_time=datetime.now(timezone.utc),
            end_time=None,
            token_range=(start_token, end_token),
            total_tokens_tested=0,
            successful_tests=0,
            failed_tests=0,
            emergency_stops=0,
            results=[]
        )
        
        self.current_session = session
        
        # Create session directory
        session_dir = self.testing_dir / session_id
        session_dir.mkdir(exist_ok=True)
        
        return session
        
    def validate_system_safety(self) -> Tuple[bool, List[str]]:
        """Comprehensive system safety validation"""
        issues = []
        
        # Check thermal status
        current_temp = self.thermal_monitor.get_current_max_temp()
        if current_temp > 85:
            issues.append(f"System temperature too high: {current_temp}°C (limit: 85°C)")
            
        # Check system resources
        memory = psutil.virtual_memory()
        if memory.percent > 85:
            issues.append(f"Memory usage too high: {memory.percent}% (limit: 85%)")
            
        cpu = psutil.cpu_percent(interval=1)
        if cpu > 80:
            issues.append(f"CPU usage too high: {cpu}% (limit: 80%)")
            
        # Check for existing DSMIL modules
        try:
            result = subprocess.run(['lsmod'], capture_output=True, text=True)
            if 'dsmil' in result.stdout:
                issues.append("DSMIL module already loaded - unload before testing")
        except Exception as e:
            issues.append(f"Could not check loaded modules: {e}")
            
        # Check for Dell SMBIOS availability
        try:
            result = subprocess.run(['which', 'smbios-token-ctl'], capture_output=True)
            if result.returncode != 0:
                issues.append("smbios-token-ctl not found - install libsmbios-bin")
        except Exception as e:
            issues.append(f"Could not verify SMBIOS tools: {e}")
            
        # Validate emergency procedures
        emergency_script = self.work_dir / "monitoring" / "emergency_stop.sh"
        if not emergency_script.exists():
            issues.append("Emergency stop script not found")
            
        return len(issues) == 0, issues
        
    def thermal_alert_callback(self, level: str, temperature: float, all_temps: List[float]):
        """Handle thermal alerts during testing"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if level == "EMERGENCY":
            print(f"\n🚨 [{timestamp}] THERMAL EMERGENCY: {temperature:.1f}°C")
            print("EMERGENCY STOP TRIGGERED!")
            self.emergency_stop()
        elif level == "CRITICAL":
            print(f"\n⚠️ [{timestamp}] THERMAL CRITICAL: {temperature:.1f}°C")
            print("Consider stopping test soon...")
        elif level == "WARNING":
            print(f"\n⚠️ [{timestamp}] THERMAL WARNING: {temperature:.1f}°C")
            
    def emergency_stop(self):
        """Emergency stop all testing operations"""
        print("\n🛑 EMERGENCY STOP ACTIVATED")
        
        self.emergency_stop_triggered = True
        self.safety_active = False
        
        # Stop thermal monitoring
        self.thermal_monitor.stop_monitoring()
        
        # Unload DSMIL module if loaded
        try:
            subprocess.run(['sudo', 'rmmod', 'dsmil-72dev'], 
                         capture_output=True, timeout=10)
            print("✓ DSMIL module unloaded")
        except Exception as e:
            print(f"⚠️ Could not unload DSMIL module: {e}")
            
        # Run emergency stop script
        emergency_script = self.work_dir / "monitoring" / "emergency_stop.sh"
        if emergency_script.exists():
            try:
                subprocess.run([str(emergency_script)], timeout=30)
                print("✓ Emergency stop script executed")
            except Exception as e:
                print(f"⚠️ Emergency script error: {e}")
                
        if self.current_session:
            self.current_session.emergency_stops += 1
            
        print("🛑 EMERGENCY STOP COMPLETE - System stabilizing...")
        
    def read_token_value(self, token: int) -> Optional[str]:
        """Read current value of SMBIOS token"""
        try:
            # Use smbios-token-ctl to read token value
            result = subprocess.run([
                'sudo', 'smbios-token-ctl', 
                '--get-token', f'{token}'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                # Parse output to extract current value
                for line in result.stdout.split('\n'):
                    if 'value:' in line.lower():
                        return line.split('=')[-1].strip()
                return result.stdout.strip()
            else:
                return None
                
        except Exception as e:
            print(f"Error reading token 0x{token:04X}: {e}")
            return None
            
    def activate_token(self, token: int, value: str = "true") -> bool:
        """Activate SMBIOS token with specified value"""
        try:
            # Activate token
            result = subprocess.run([
                'sudo', 'smbios-token-ctl',
                '--set-token', f'{token}',
                '--value', value
            ], capture_output=True, text=True, timeout=15)
            
            return result.returncode == 0
            
        except Exception as e:
            print(f"Error activating token 0x{token:04X}: {e}")
            return False
            
    def check_dsmil_response(self) -> List[str]:
        """Check for DSMIL kernel module responses"""
        dsmil_messages = []
        
        try:
            # Check dmesg for DSMIL-related messages
            result = subprocess.run([
                'dmesg', '--time-format=iso'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines[-50:]:  # Check last 50 lines
                    if 'dsmil' in line.lower():
                        dsmil_messages.append(line)
                        
            # Check for DSMIL device responses
            dsmil_sysfs = Path("/sys/module/dsmil_72dev")
            if dsmil_sysfs.exists():
                # Look for device status changes
                for device_dir in dsmil_sysfs.glob("devices/*/"):
                    status_file = device_dir / "status"
                    if status_file.exists():
                        try:
                            status = status_file.read_text().strip()
                            if status != "inactive":
                                dsmil_messages.append(f"Device {device_dir.name}: {status}")
                        except OSError:
                            pass
                            
        except Exception as e:
            print(f"Error checking DSMIL response: {e}")
            
        return dsmil_messages
        
    def rollback_token(self, token: int, original_value: str) -> bool:
        """Rollback token to original value"""
        try:
            if original_value and original_value != "unknown":
                result = subprocess.run([
                    'sudo', 'smbios-token-ctl',
                    '--set-token', f'{token}',
                    '--value', original_value
                ], capture_output=True, text=True, timeout=15)
                
                return result.returncode == 0
            else:
                # If original value unknown, try common safe values
                for safe_value in ["false", "0", "disable"]:
                    result = subprocess.run([
                        'sudo', 'smbios-token-ctl',
                        '--set-token', f'{token}',
                        '--value', safe_value
                    ], capture_output=True, text=True, timeout=15)
                    
                    if result.returncode == 0:
                        return True
                        
                return False
                
        except Exception as e:
            print(f"Error rolling back token 0x{token:04X}: {e}")
            return False
            
    def test_single_token(self, token: int, activation_value: str = "true", 
                         test_duration: int = 5) -> TokenTestResult:
        """Test a single SMBIOS token with full safety protocol"""
        
        print(f"\n🧪 Testing token 0x{token:04X}...")
        
        test_start = datetime.now(timezone.utc)
        errors = []
        warnings = []
        thermal_readings = []
        
        # Pre-test safety check
        current_temp = self.thermal_monitor.get_current_max_temp()
        if current_temp > 90:
            warnings.append(f"High temperature before test: {current_temp:.1f}°C")
            
        # Read initial value
        initial_value = self.read_token_value(token)
        print(f"  Initial value: {initial_value}")
        
        # Activate token
        print(f"  Activating with value: {activation_value}")
        activation_successful = self.activate_token(token, activation_value)
        
        if not activation_successful:
            errors.append("Token activation failed")
            
        # Monitor for specified duration
        print(f"  Monitoring for {test_duration} seconds...")
        dsmil_responses = []
        
        for i in range(test_duration):
            if self.emergency_stop_triggered:
                errors.append("Emergency stop triggered during test")
                break
                
            # Check thermal
            temp = self.thermal_monitor.get_current_max_temp()
            thermal_readings.append(temp)
            
            # Check DSMIL response
            responses = self.check_dsmil_response()
            dsmil_responses.extend(responses)
            
            # Progress indicator
            if i % 2 == 0:
                print(f"    [{i+1}/{test_duration}] Temp: {temp:.1f}°C")
                
            time.sleep(1)
            
        # Read final value
        final_value = self.read_token_value(token)
        print(f"  Final value: {final_value}")
        
        # Rollback to original value
        print(f"  Rolling back to: {initial_value}")
        rollback_successful = self.rollback_token(token, initial_value or "false")
        
        if not rollback_successful:
            errors.append("Token rollback failed")
            warnings.append("Manual rollback may be required")
            
        # Check post-rollback value
        post_rollback_value = self.read_token_value(token)
        if post_rollback_value != initial_value:
            warnings.append(f"Rollback verification failed: {post_rollback_value} != {initial_value}")
            
        # Collect system impact data
        system_impact = {
            'max_temperature': max(thermal_readings) if thermal_readings else 0,
            'avg_temperature': sum(thermal_readings) / len(thermal_readings) if thermal_readings else 0,
            'temperature_change': (max(thermal_readings) - thermal_readings[0]) if len(thermal_readings) > 1 else 0,
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'dsmil_responses_count': len(set(dsmil_responses))
        }
        
        result = TokenTestResult(
            token=f"0x{token:04X}",
            test_time=test_start,
            initial_value=initial_value,
            activation_successful=activation_successful,
            final_value=final_value,
            dsmil_response=list(set(dsmil_responses)),  # Remove duplicates
            thermal_readings=thermal_readings,
            rollback_successful=rollback_successful,
            errors=errors,
            warnings=warnings,
            system_impact=system_impact
        )
        
        # Update session stats
        if self.current_session:
            self.current_session.results.append(result)
            self.current_session.total_tokens_tested += 1
            if not errors:
                self.current_session.successful_tests += 1
            else:
                self.current_session.failed_tests += 1
                
        status = "✅ PASS" if not errors else "❌ FAIL"
        print(f"  Result: {status}")
        
        if warnings:
            print(f"  Warnings: {len(warnings)}")
        if errors:
            print(f"  Errors: {len(errors)}")
            
        return result
        
    def test_token_group(self, group_name: str, delay_between_tests: int = 10) -> List[TokenTestResult]:
        """Test a complete group of 12 DSMIL tokens"""
        
        if group_name not in self.DSMIL_GROUPS:
            raise ValueError(f"Unknown group: {group_name}")
            
        tokens = self.DSMIL_GROUPS[group_name]
        print(f"\n🧪 Testing {group_name}: {len(tokens)} tokens")
        print(f"   Range: 0x{tokens[0]:04X} - 0x{tokens[-1]:04X}")
        
        group_results = []
        
        for i, token in enumerate(tokens, 1):
            if self.emergency_stop_triggered:
                print(f"\n🛑 Emergency stop - aborting group test at token {i}/{len(tokens)}")
                break
                
            print(f"\n--- Token {i}/{len(tokens)} in {group_name} ---")
            
            # Safety validation before each test
            is_safe, issues = self.validate_system_safety()
            if not is_safe:
                print(f"⚠️ Safety check failed: {issues}")
                print("Skipping token test")
                continue
                
            result = self.test_single_token(token)
            group_results.append(result)
            
            # Inter-test delay
            if i < len(tokens) and delay_between_tests > 0:
                print(f"Waiting {delay_between_tests} seconds before next test...")
                time.sleep(delay_between_tests)
                
        return group_results
        
    def save_test_results(self, filename: Optional[str] = None):
        """Save test results to JSON file"""
        
        if not self.current_session:
            print("No active test session to save")
            return
            
        if filename is None:
            filename = f"{self.current_session.session_id}_results.json"
            
        filepath = self.testing_dir / filename
        
        # Convert session to dict for JSON serialization
        session_dict = {
            'session_id': self.current_session.session_id,
            'start_time': self.current_session.start_time.isoformat(),
            'end_time': self.current_session.end_time.isoformat() if self.current_session.end_time else None,
            'token_range': self.current_session.token_range,
            'total_tokens_tested': self.current_session.total_tokens_tested,
            'successful_tests': self.current_session.successful_tests,
            'failed_tests': self.current_session.failed_tests,
            'emergency_stops': self.current_session.emergency_stops,
            'results': []
        }
        
        # Convert test results
        for result in self.current_session.results:
            result_dict = {
                'token': result.token,
                'test_time': result.test_time.isoformat(),
                'initial_value': result.initial_value,
                'activation_successful': result.activation_successful,
                'final_value': result.final_value,
                'dsmil_response': result.dsmil_response,
                'thermal_readings': result.thermal_readings,
                'rollback_successful': result.rollback_successful,
                'errors': result.errors,
                'warnings': result.warnings,
                'system_impact': result.system_impact
            }
            session_dict['results'].append(result_dict)
            
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(session_dict, f, indent=2, default=str)
            
        print(f"📊 Test results saved to: {filepath}")
        
    def generate_test_report(self) -> str:
        """Generate comprehensive test report"""
        
        if not self.current_session:
            return "No active test session"
            
        report = []
        report.append("=" * 80)
        report.append("SMBIOS TOKEN TESTING REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Session summary
        report.append(f"Session ID: {self.current_session.session_id}")
        report.append(f"Start Time: {self.current_session.start_time}")
        report.append(f"Token Range: {self.current_session.token_range[0]} - {self.current_session.token_range[1]}")
        report.append(f"Total Tests: {self.current_session.total_tokens_tested}")
        report.append(f"Successful: {self.current_session.successful_tests}")
        report.append(f"Failed: {self.current_session.failed_tests}")
        report.append(f"Emergency Stops: {self.current_session.emergency_stops}")
        report.append("")
        
        # Individual test results
        for result in self.current_session.results:
            report.append(f"Token {result.token}:")
            report.append(f"  Status: {'PASS' if not result.errors else 'FAIL'}")
            report.append(f"  Initial Value: {result.initial_value}")
            report.append(f"  Final Value: {result.final_value}")
            report.append(f"  Max Temperature: {max(result.thermal_readings):.1f}°C")
            report.append(f"  DSMIL Responses: {len(result.dsmil_response)}")
            
            if result.errors:
                report.append(f"  Errors: {', '.join(result.errors)}")
            if result.warnings:
                report.append(f"  Warnings: {', '.join(result.warnings)}")
                
            report.append("")
            
        return "\n".join(report)

def main():
    """Main testing interface"""
    
    print("🧪 SMBIOS Token Testing Framework v1.0.0")
    print("Dell Latitude 5450 MIL-SPEC - TESTBED Agent")
    print("=" * 60)
    
    tester = SMBIOSTokenTester()
    
    # System safety validation
    print("\n🔒 Validating system safety...")
    is_safe, issues = tester.validate_system_safety()
    
    if not is_safe:
        print("❌ System safety validation failed:")
        for issue in issues:
            print(f"  ⚠️ {issue}")
        print("\nPlease resolve issues before testing.")
        return 1
        
    print("✅ System safety validation passed")
    
    # Start thermal monitoring
    print("\n🌡️ Starting thermal monitoring...")
    tester.thermal_monitor.start_monitoring(tester.thermal_alert_callback)
    
    try:
        # Create test session
        session = tester.create_test_session("Range_0480")
        print(f"\n📋 Created test session: {session.session_id}")
        
        # Test options
        print("\nTesting Options:")
        print("1. Test single token (0x0480)")
        print("2. Test Group_0 (12 tokens: 0x0480-0x048B)")
        print("3. Test all groups (72 tokens: 0x0480-0x04C7)")
        print("4. Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "1":
            # Single token test
            result = tester.test_single_token(0x0480)
            print(f"\nTest completed: {'PASS' if not result.errors else 'FAIL'}")
            
        elif choice == "2":
            # Group test
            print("\n⚠️ This will test 12 tokens (approximately 5 minutes)")
            confirm = input("Proceed? (y/N): ").strip().lower()
            if confirm == 'y':
                results = tester.test_token_group("Group_0")
                print(f"\nGroup test completed: {len(results)} tokens tested")
                
        elif choice == "3":
            # Full testing (all groups)
            print("\n⚠️ This will test 72 tokens (approximately 30 minutes)")
            confirm = input("Proceed? (y/N): ").strip().lower()
            if confirm == 'y':
                for group_name in tester.DSMIL_GROUPS:
                    if tester.emergency_stop_triggered:
                        break
                    print(f"\n=== Testing {group_name} ===")
                    results = tester.test_token_group(group_name, delay_between_tests=15)
                    print(f"{group_name} completed: {len(results)} tokens")
                    
        # Finalize session
        tester.current_session.end_time = datetime.now(timezone.utc)
        
        # Save results
        tester.save_test_results()
        
        # Generate report
        report = tester.generate_test_report()
        print("\n" + report)
        
        # Save report to file
        report_file = tester.testing_dir / f"{session.session_id}_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"📊 Report saved to: {report_file}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Testing interrupted by user")
        tester.emergency_stop()
        
    except Exception as e:
        print(f"\n❌ Testing error: {e}")
        tester.emergency_stop()
        return 1
        
    finally:
        # Stop thermal monitoring
        print("\n🛑 Stopping thermal monitoring...")
        tester.thermal_monitor.stop_monitoring()
        
    print("\n✅ Testing framework shutdown complete")
    return 0

if __name__ == "__main__":
    sys.exit(main())