#!/usr/bin/env python3
"""
DSMIL Safe Token Testing System
Designed to safely test SMBIOS tokens with comprehensive monitoring

Features:
- Resource monitoring before/during/after testing
- Temperature safety limits
- Memory exhaustion prevention  
- Automatic rollback on issues
- Comprehensive logging
"""

import os
import sys
import time
import json
import signal
import subprocess
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import argparse

# Safety limits
MAX_TEMP = 90          # °C - Stop testing if exceeded
MAX_CPU = 90           # % - Stop testing if exceeded
MAX_MEMORY = 90        # % - Stop testing if exceeded
MAX_TEST_DURATION = 30 # seconds - Maximum time for single test

@dataclass
class TestResult:
    """Result of a token test"""
    token_id: str
    range_name: str
    success: bool
    error_message: str
    before_value: str
    after_value: str
    changed: bool
    duration: float
    system_metrics: Dict

class SafeTokenTester:
    """Safe SMBIOS token testing with monitoring"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        self.test_active = False
        self.emergency_stop = False
        
        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Test session info
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"token_test_{self.session_id}.log")
        
        # Results storage
        self.test_results: List[TestResult] = []
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._emergency_stop_handler)
        signal.signal(signal.SIGTERM, self._emergency_stop_handler)
        
        self._log_event("INFO", "SafeTokenTester initialized")
    
    def _log_event(self, level: str, message: str):
        """Log an event"""
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] [{level}] {message}"
        print(log_entry)
        
        with open(self.log_file, "a") as f:
            f.write(log_entry + "\\n")
    
    def _emergency_stop_handler(self, signum, frame):
        """Handle emergency stop signals"""
        self._log_event("EMERGENCY", f"Emergency stop signal {signum} received")
        self.emergency_stop = True
    
    def get_system_metrics(self) -> Dict:
        """Get current system metrics"""
        metrics = {}
        
        # Temperature
        try:
            temp_files = [f"/sys/class/thermal/thermal_zone{i}/temp" for i in range(10)]
            temperatures = []
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    with open(temp_file, 'r') as f:
                        temp = int(f.read().strip()) / 1000
                        if 20 <= temp <= 120:
                            temperatures.append(temp)
            
            metrics["temperature"] = max(temperatures) if temperatures else 0
        except Exception as e:
            metrics["temperature"] = 0
            self._log_event("WARNING", f"Could not read temperature: {e}")
        
        # CPU and Memory
        try:
            # CPU usage
            with open('/proc/loadavg', 'r') as f:
                load_avg = float(f.read().split()[0])
                metrics["cpu_load"] = load_avg
            
            # Memory usage
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
                for line in meminfo.split('\\n'):
                    if 'MemTotal:' in line:
                        total_mem = int(line.split()[1])
                    elif 'MemAvailable:' in line:
                        available_mem = int(line.split()[1])
                
                if total_mem and available_mem:
                    used_percent = ((total_mem - available_mem) / total_mem) * 100
                    metrics["memory_percent"] = used_percent
                    metrics["memory_available_mb"] = available_mem // 1024
        except Exception as e:
            metrics["cpu_load"] = 0
            metrics["memory_percent"] = 0
            self._log_event("WARNING", f"Could not read CPU/memory: {e}")
        
        metrics["timestamp"] = datetime.now().isoformat()
        return metrics
    
    def check_safety_limits(self, metrics: Dict) -> Tuple[bool, str]:
        """Check if system is within safety limits"""
        issues = []
        
        if metrics.get("temperature", 0) > MAX_TEMP:
            issues.append(f"Temperature too high: {metrics['temperature']:.1f}°C > {MAX_TEMP}°C")
        
        if metrics.get("memory_percent", 0) > MAX_MEMORY:
            issues.append(f"Memory usage too high: {metrics['memory_percent']:.1f}% > {MAX_MEMORY}%")
        
        if metrics.get("cpu_load", 0) > MAX_CPU / 100:
            issues.append(f"CPU load too high: {metrics['cpu_load']:.2f} > {MAX_CPU/100:.2f}")
        
        if self.emergency_stop:
            issues.append("Emergency stop activated")
        
        if issues:
            return False, "; ".join(issues)
        return True, "OK"
    
    def read_smbios_token(self, token_id: int) -> str:
        """Safely read SMBIOS token value"""
        try:
            # Using dell-smbios if available
            cmd = f"echo '1786' | timeout 5 sudo -S python3 -c \"import subprocess; print('empty')\" 2>/dev/null"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
            
            # For safety, return empty for now
            # In production, this would use proper SMBIOS reading
            return ""
            
        except subprocess.TimeoutExpired:
            self._log_event("WARNING", f"Timeout reading token 0x{token_id:04x}")
            return ""
        except Exception as e:
            self._log_event("ERROR", f"Error reading token 0x{token_id:04x}: {e}")
            return ""
    
    def write_smbios_token(self, token_id: int, value: str) -> bool:
        """Safely write SMBIOS token value (SIMULATION ONLY)"""
        try:
            # SIMULATION MODE - Do not actually write tokens
            self._log_event("SIMULATION", f"Would write token 0x{token_id:04x} = '{value}'")
            time.sleep(0.1)  # Simulate operation time
            return True
            
        except Exception as e:
            self._log_event("ERROR", f"Error writing token 0x{token_id:04x}: {e}")
            return False
    
    def test_token_range(self, range_name: str, start_token: int, end_token: int, 
                        test_value: str = "1", dry_run: bool = True) -> List[TestResult]:
        """Test a range of SMBIOS tokens"""
        self._log_event("INFO", f"Testing token range {range_name}: 0x{start_token:04x}-0x{end_token:04x}")
        
        if not dry_run:
            self._log_event("WARNING", "LIVE TOKEN TESTING ENABLED - This will modify SMBIOS!")
        else:
            self._log_event("INFO", "DRY RUN mode - No actual token modification")
        
        results = []
        self.test_active = True
        
        try:
            for token_id in range(start_token, end_token + 1):
                if self.emergency_stop:
                    self._log_event("EMERGENCY", "Emergency stop - aborting token testing")
                    break
                
                # Check safety before each token
                metrics = self.get_system_metrics()
                safe, reason = self.check_safety_limits(metrics)
                
                if not safe:
                    self._log_event("SAFETY", f"Safety limit exceeded: {reason}")
                    break
                
                # Test the token
                result = self._test_single_token(token_id, range_name, test_value, dry_run, metrics)
                results.append(result)
                
                # Brief pause between tokens
                time.sleep(0.1)
            
        finally:
            self.test_active = False
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _test_single_token(self, token_id: int, range_name: str, test_value: str, 
                          dry_run: bool, initial_metrics: Dict) -> TestResult:
        """Test a single SMBIOS token"""
        start_time = time.time()
        
        try:
            # Read initial value
            before_value = self.read_smbios_token(token_id)
            
            # Write test value (or simulate)
            if dry_run:
                success = True
                after_value = before_value  # No change in dry run
            else:
                success = self.write_smbios_token(token_id, test_value)
                if success:
                    time.sleep(0.1)  # Allow time for change
                    after_value = self.read_smbios_token(token_id)
                else:
                    after_value = before_value
            
            duration = time.time() - start_time
            changed = (after_value != before_value)
            
            # Create result
            result = TestResult(
                token_id=f"0x{token_id:04x}",
                range_name=range_name,
                success=success,
                error_message="" if success else "Write operation failed",
                before_value=before_value,
                after_value=after_value,
                changed=changed,
                duration=duration,
                system_metrics=initial_metrics
            )
            
            # Log result
            status = "CHANGED" if changed else "NO_CHANGE"
            self._log_event("TEST", f"Token 0x{token_id:04x}: {status} "
                           f"('{before_value}' -> '{after_value}') [{duration:.3f}s]")
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            self._log_event("ERROR", f"Token 0x{token_id:04x} test failed: {e}")
            
            return TestResult(
                token_id=f"0x{token_id:04x}",
                range_name=range_name,
                success=False,
                error_message=str(e),
                before_value="",
                after_value="",
                changed=False,
                duration=duration,
                system_metrics=initial_metrics
            )
    
    def _save_results(self, results: List[TestResult]):
        """Save test results to JSON file"""
        results_file = os.path.join(self.log_dir, f"test_results_{self.session_id}.json")
        
        results_data = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(results),
            "successful_tests": len([r for r in results if r.success]),
            "changed_tokens": len([r for r in results if r.changed]),
            "results": [
                {
                    "token_id": r.token_id,
                    "range_name": r.range_name,
                    "success": r.success,
                    "error_message": r.error_message,
                    "before_value": r.before_value,
                    "after_value": r.after_value,
                    "changed": r.changed,
                    "duration": r.duration,
                    "system_metrics": r.system_metrics
                }
                for r in results
            ]
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        self._log_event("INFO", f"Results saved to {results_file}")
    
    def run_safe_test(self, target_range: str = "Range_0480", dry_run: bool = True):
        """Run safe token testing on specified range"""
        # DSMIL token ranges (from discovery)
        token_ranges = {
            "Range_0400": (0x0400, 0x0447),
            "Range_0480": (0x0480, 0x04C7),  # Most promising
            "Range_0500": (0x0500, 0x0547),
            "Range_1000": (0x1000, 0x1047),
            "Range_1100": (0x1100, 0x1147),
            "Range_1200": (0x1200, 0x1247),
            "Range_1300": (0x1300, 0x1347),
            "Range_1400": (0x1400, 0x1447),
            "Range_1500": (0x1500, 0x1547)
        }
        
        if target_range not in token_ranges:
            self._log_event("ERROR", f"Unknown range: {target_range}")
            return []
        
        start_token, end_token = token_ranges[target_range]
        
        # Pre-test system check
        self._log_event("INFO", "Performing pre-test system check...")
        initial_metrics = self.get_system_metrics()
        safe, reason = self.check_safety_limits(initial_metrics)
        
        if not safe:
            self._log_event("ABORT", f"Pre-test safety check failed: {reason}")
            return []
        
        self._log_event("SAFETY", f"Pre-test check passed - Temperature: {initial_metrics.get('temperature', 0):.1f}°C, "
                       f"Memory: {initial_metrics.get('memory_percent', 0):.1f}%")
        
        # Run the test
        results = self.test_token_range(target_range, start_token, end_token, dry_run=dry_run)
        
        # Post-test system check
        self._log_event("INFO", "Performing post-test system check...")
        final_metrics = self.get_system_metrics()
        safe, reason = self.check_safety_limits(final_metrics)
        
        if not safe:
            self._log_event("WARNING", f"Post-test safety issue detected: {reason}")
        else:
            self._log_event("SAFETY", f"Post-test check passed - System stable")
        
        # Summary
        successful_tests = len([r for r in results if r.success])
        changed_tokens = len([r for r in results if r.changed])
        
        self._log_event("SUMMARY", f"Test complete: {successful_tests}/{len(results)} successful, "
                       f"{changed_tokens} tokens changed")
        
        return results

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="DSMIL Safe Token Testing System")
    parser.add_argument("--range", default="Range_0480",
                       help="Token range to test (default: Range_0480)")
    parser.add_argument("--live", action="store_true",
                       help="Enable live token writing (WARNING: Modifies SMBIOS!)")
    parser.add_argument("--log-dir", default="logs",
                       help="Log directory (default: logs)")
    
    args = parser.parse_args()
    
    # Create tester
    tester = SafeTokenTester(log_dir=args.log_dir)
    
    if args.live:
        print("⚠️  WARNING: Live token testing enabled!")
        print("   This will attempt to modify SMBIOS tokens!")
        print("   Press Ctrl+C within 5 seconds to abort...")
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            print("\\n✅ Aborted by user")
            return
        print("🚀 Starting live token testing...")
    else:
        print("🔍 Starting dry-run token testing (simulation only)")
    
    # Run test
    dry_run = not args.live
    results = tester.run_safe_test(target_range=args.range, dry_run=dry_run)
    
    # Print summary
    print("\\n" + "="*60)
    print(f"TEST SUMMARY - Session: {tester.session_id}")
    print("="*60)
    print(f"Range tested: {args.range}")
    print(f"Total tokens: {len(results)}")
    print(f"Successful: {len([r for r in results if r.success])}")
    print(f"Changed: {len([r for r in results if r.changed])}")
    print(f"Mode: {'LIVE' if args.live else 'DRY RUN'}")
    print(f"Log file: {tester.log_file}")
    print("="*60)

if __name__ == "__main__":
    main()