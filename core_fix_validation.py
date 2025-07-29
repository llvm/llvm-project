#!/usr/bin/env python3
"""
Comprehensive validation tests for LLDB-DAP core performance fixes.
This validates that the core fixes work reliably across different scenarios.
"""

import subprocess
import time
import json
import os
import sys
import tempfile
from pathlib import Path

class CoreFixValidator:
    def __init__(self, lldb_dap_path):
        self.lldb_dap_path = lldb_dap_path
        self.test_results = {}
        
    def create_test_program(self, name="test_program"):
        """Create a test program with debug info."""
        test_file = Path(f"{name}.c")
        test_file.write_text(f"""
#include <stdio.h>
#include <unistd.h>

int main() {{
    printf("Hello from {name}\\n");
    sleep(1);  // Give time for debugger to attach
    return 0;
}}
""")
        
        # Compile with debug info
        subprocess.run(["clang", "-g", "-o", name, str(test_file)], check=True)
        return Path(name).absolute()

    def test_performance_regression(self):
        """Test that launch times are under 500ms consistently."""
        print("=== Testing Performance Regression ===")
        
        program = self.create_test_program("perf_test")
        times = []
        
        for i in range(5):
            start_time = time.time()
            
            try:
                process = subprocess.Popen(
                    [str(self.lldb_dap_path)],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # Send minimal DAP sequence
                init_msg = self._create_dap_message("initialize",
                                                   {"clientID": "test"})
                launch_msg = self._create_dap_message("launch", {
                    "program": str(program),
                    "stopOnEntry": True
                })
                
                process.stdin.write(init_msg)
                process.stdin.write(launch_msg)
                process.stdin.flush()
                
                # Wait for response or timeout
                stdout, stderr = process.communicate(timeout=5)
                end_time = time.time()
                
                duration = (end_time - start_time) * 1000
                times.append(duration)
                print(f"  Run {i+1}: {duration:.1f}ms")
                
            except subprocess.TimeoutExpired:
                process.kill()
                times.append(5000)  # Timeout
                print(f"  Run {i+1}: TIMEOUT")
        
        avg_time = sum(times) / len(times)
        max_time = max(times)
        
        # Validate performance requirements
        performance_ok = avg_time < 500 and max_time < 1000
        
        self.test_results['performance_regression'] = {
            'passed': performance_ok,
            'avg_time_ms': avg_time,
            'max_time_ms': max_time,
            'times': times,
            'requirement': 'avg < 500ms, max < 1000ms'
        }
        
        print(f"  Average: {avg_time:.1f}ms, Max: {max_time:.1f}ms")
        print(f"  Result: {'PASS' if performance_ok else 'FAIL'}")
        
        return performance_ok

    def test_network_symbol_scenarios(self):
        """Test behavior with different network conditions."""
        print("=== Testing Network Symbol Scenarios ===")
        
        program = self.create_test_program("network_test")
        scenarios = [
            ("no_debuginfod", {}),
            ("with_debuginfod",
             {"DEBUGINFOD_URLS": "http://debuginfod.example.com"}),
            ("slow_debuginfod",
             {"DEBUGINFOD_URLS": "http://slow.debuginfod.example.com"}),
        ]
        
        results = {}
        
        for scenario_name, env_vars in scenarios:
            print(f"  Testing {scenario_name}...")
            
            # Set up environment
            test_env = os.environ.copy()
            test_env.update(env_vars)
            
            start_time = time.time()
            
            try:
                process = subprocess.Popen(
                    [str(self.lldb_dap_path)],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=test_env
                )
                
                init_msg = self._create_dap_message("initialize", {"clientID": "test"})
                launch_msg = self._create_dap_message("launch", {
                    "program": str(program),
                    "stopOnEntry": True
                })
                
                process.stdin.write(init_msg)
                process.stdin.write(launch_msg)
                process.stdin.flush()
                
                stdout, stderr = process.communicate(timeout=10)
                end_time = time.time()
                
                duration = (end_time - start_time) * 1000
                results[scenario_name] = {
                    'duration_ms': duration,
                    'success': True,
                    'timeout': False
                }
                
                print(f"    {scenario_name}: {duration:.1f}ms - SUCCESS")
                
            except subprocess.TimeoutExpired:
                process.kill()
                results[scenario_name] = {
                    'duration_ms': 10000,
                    'success': False,
                    'timeout': True
                }
                print(f"    {scenario_name}: TIMEOUT - FAIL")
        
        # Validate that all scenarios complete reasonably quickly
        all_passed = all(r['duration_ms'] < 3000 for r in results.values())
        
        self.test_results['network_scenarios'] = {
            'passed': all_passed,
            'scenarios': results
        }
        
        print(f"  Overall: {'PASS' if all_passed else 'FAIL'}")
        return all_passed

    def test_cross_platform_performance(self):
        """Test performance consistency across different conditions."""
        print("=== Testing Cross-Platform Performance ===")
        
        # Test with different program sizes
        test_cases = [
            ("small", self._create_small_program),
            ("medium", self._create_medium_program),
        ]
        
        results = {}
        
        for case_name, program_creator in test_cases:
            print(f"  Testing {case_name} program...")
            
            program = program_creator()
            times = []
            
            for i in range(3):
                start_time = time.time()
                
                try:
                    process = subprocess.Popen(
                        [str(self.lldb_dap_path)],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    
                    init_msg = self._create_dap_message("initialize", {"clientID": "test"})
                    launch_msg = self._create_dap_message("launch", {
                        "program": str(program),
                        "stopOnEntry": True
                    })
                    
                    process.stdin.write(init_msg)
                    process.stdin.write(launch_msg)
                    process.stdin.flush()
                    
                    stdout, stderr = process.communicate(timeout=5)
                    end_time = time.time()
                    
                    duration = (end_time - start_time) * 1000
                    times.append(duration)
                    
                except subprocess.TimeoutExpired:
                    process.kill()
                    times.append(5000)
            
            avg_time = sum(times) / len(times)
            results[case_name] = {
                'avg_time_ms': avg_time,
                'times': times,
                'passed': avg_time < 1000
            }
            
            print(f"    {case_name}: {avg_time:.1f}ms avg - {'PASS' if avg_time < 1000 else 'FAIL'}")
        
        all_passed = all(r['passed'] for r in results.values())
        
        self.test_results['cross_platform'] = {
            'passed': all_passed,
            'cases': results
        }
        
        return all_passed

    def _create_small_program(self):
        """Create a small test program."""
        return self.create_test_program("small_test")

    def _create_medium_program(self):
        """Create a medium-sized test program with more symbols."""
        test_file = Path("medium_test.c")
        test_file.write_text("""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct TestStruct {
    int value;
    char name[64];
    double data[100];
};

void function1() { printf("Function 1\\n"); }
void function2() { printf("Function 2\\n"); }
void function3() { printf("Function 3\\n"); }

int main() {
    struct TestStruct test;
    test.value = 42;
    strcpy(test.name, "test");
    
    for (int i = 0; i < 100; i++) {
        test.data[i] = i * 3.14;
    }
    
    function1();
    function2();
    function3();
    
    return 0;
}
""")
        
        subprocess.run(["clang", "-g", "-o", "medium_test", str(test_file)], check=True)
        return Path("medium_test").absolute()

    def _create_dap_message(self, command, arguments=None):
        """Create a DAP protocol message."""
        if arguments is None:
            arguments = {}
        
        message = {
            "seq": 1,
            "type": "request",
            "command": command,
            "arguments": arguments
        }
        
        content = json.dumps(message)
        return f"Content-Length: {len(content)}\r\n\r\n{content}"

    def run_all_tests(self):
        """Run all validation tests."""
        print("LLDB-DAP Core Fix Validation")
        print("=" * 50)
        
        tests = [
            ("Performance Regression", self.test_performance_regression),
            ("Network Symbol Scenarios", self.test_network_symbol_scenarios),
            ("Cross-Platform Performance", self.test_cross_platform_performance),
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            try:
                if test_func():
                    passed_tests += 1
                print()
            except Exception as e:
                print(f"  ERROR: {e}")
                print()
        
        # Summary
        print("=" * 50)
        print("VALIDATION SUMMARY:")
        print("=" * 50)
        print(f"Tests passed: {passed_tests}/{total_tests}")
        
        for test_name, result in self.test_results.items():
            status = "PASS" if result['passed'] else "FAIL"
            print(f"{test_name:25}: {status}")
        
        overall_success = passed_tests == total_tests
        print(f"\nOverall result: {'SUCCESS' if overall_success else 'FAILURE'}")
        
        return overall_success

def main():
    """Main validation function."""
    lldb_dap_path = Path("./build/bin/lldb-dap")
    
    if not lldb_dap_path.exists():
        print(f"Error: lldb-dap not found at {lldb_dap_path}")
        return 1
    
    validator = CoreFixValidator(lldb_dap_path)
    success = validator.run_all_tests()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
