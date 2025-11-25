#!/usr/bin/env python3
"""
SMBIOS Token Testing Orchestration System
==========================================

High-level orchestration system for Dell Latitude 5450 MIL-SPEC SMBIOS token testing.
Coordinates all testing components and provides unified interface for comprehensive
token testing campaigns.

Features:
- Automated safety validation before testing
- Coordinated multi-system testing (monitoring, validation, testing)
- Real-time progress tracking and reporting
- Emergency response coordination
- Test session management with comprehensive logging
- Debian Trixie and Ubuntu 24.04 compatibility

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
import signal
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

# Import our testing components
sys.path.append(str(Path(__file__).parent))
from smbios_testbed_framework import SMBIOSTokenTester, TestSession, TokenTestResult
from safety_validator import SafetyValidator, SafetyReport, SafetyLevel
from debian_compatibility import DebianCompatibility

class TestingPhase(Enum):
    """Testing phase enumeration"""
    INITIALIZATION = "INITIALIZATION"
    SAFETY_VALIDATION = "SAFETY_VALIDATION"
    SYSTEM_PREPARATION = "SYSTEM_PREPARATION"
    TOKEN_TESTING = "TOKEN_TESTING"
    RESULTS_ANALYSIS = "RESULTS_ANALYSIS"
    CLEANUP = "CLEANUP"
    COMPLETED = "COMPLETED"
    ABORTED = "ABORTED"

@dataclass
class TestingCampaign:
    """Complete testing campaign information"""
    campaign_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    phase: TestingPhase = TestingPhase.INITIALIZATION
    target_ranges: List[str] = field(default_factory=list)
    completed_ranges: List[str] = field(default_factory=list)
    failed_ranges: List[str] = field(default_factory=list)
    total_tokens_tested: int = 0
    successful_tests: int = 0
    failed_tests: int = 0
    emergency_stops: int = 0
    safety_reports: List[str] = field(default_factory=list)
    test_sessions: List[str] = field(default_factory=list)
    status_log: List[str] = field(default_factory=list)

class TokenTestingOrchestrator:
    """Main orchestration system for SMBIOS token testing"""
    
    def __init__(self, work_dir: str = "/home/john/LAT5150DRVMIL"):
        self.work_dir = Path(work_dir)
        self.testing_dir = self.work_dir / "testing"
        self.testing_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.tester = SMBIOSTokenTester(str(self.work_dir))
        self.safety_validator = SafetyValidator(str(self.work_dir))
        self.debian_compat = DebianCompatibility()
        
        # Campaign management
        self.current_campaign: Optional[TestingCampaign] = None
        self.emergency_stop_triggered = False
        self.monitoring_processes: List[subprocess.Popen] = []
        
        # Signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Pre-defined testing scenarios
        self.testing_scenarios = {
            'single_token': {
                'name': 'Single Token Test',
                'description': 'Test a single SMBIOS token for initial validation',
                'ranges': ['single_0x0480'],
                'estimated_time': '2 minutes'
            },
            'group_test': {
                'name': 'Group Test',
                'description': 'Test one group of 12 DSMIL tokens',
                'ranges': ['Group_0'],
                'estimated_time': '10 minutes'
            },
            'range_0480': {
                'name': 'Range 0x0480 Full Test',
                'description': 'Test all 72 tokens in Range 0x0480-0x04C7',
                'ranges': ['Range_0480'],
                'estimated_time': '45 minutes'
            },
            'comprehensive': {
                'name': 'Comprehensive Test',
                'description': 'Test all available token ranges',
                'ranges': ['Range_0480', 'Range_0400', 'Range_0500'],
                'estimated_time': '2 hours'
            }
        }
        
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully"""
        print(f"\n⚠️ Received signal {signum} - initiating graceful shutdown...")
        self.emergency_stop_triggered = True
        self._emergency_stop()
        
    def _log_status(self, message: str, level: str = "INFO"):
        """Log status message with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        level_emoji = {
            "INFO": "ℹ️",
            "WARNING": "⚠️", 
            "ERROR": "❌",
            "SUCCESS": "✅",
            "CRITICAL": "🚨"
        }
        
        emoji = level_emoji.get(level, "ℹ️")
        log_message = f"[{timestamp}] {emoji} {message}"
        print(log_message)
        
        if self.current_campaign:
            self.current_campaign.status_log.append(log_message)
            
    def create_testing_campaign(self, scenario: str, custom_ranges: List[str] = None) -> TestingCampaign:
        """Create a new testing campaign"""
        
        campaign_id = f"campaign_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if custom_ranges:
            target_ranges = custom_ranges
        elif scenario in self.testing_scenarios:
            target_ranges = self.testing_scenarios[scenario]['ranges']
        else:
            raise ValueError(f"Unknown testing scenario: {scenario}")
            
        campaign = TestingCampaign(
            campaign_id=campaign_id,
            start_time=datetime.now(timezone.utc),
            target_ranges=target_ranges
        )
        
        self.current_campaign = campaign
        
        # Create campaign directory
        campaign_dir = self.testing_dir / campaign_id
        campaign_dir.mkdir(exist_ok=True)
        
        self._log_status(f"Created testing campaign: {campaign_id}")
        self._log_status(f"Target ranges: {', '.join(target_ranges)}")
        
        return campaign
        
    def run_pre_test_validation(self) -> Tuple[bool, SafetyReport]:
        """Run comprehensive pre-test validation"""
        
        self._log_status("Starting pre-test validation...")
        
        # Update campaign phase
        if self.current_campaign:
            self.current_campaign.phase = TestingPhase.SAFETY_VALIDATION
            
        # Check distribution compatibility
        self._log_status("Checking distribution compatibility...")
        is_compatible, issues = self.debian_compat.check_system_compatibility()
        
        if not is_compatible:
            self._log_status("Distribution compatibility issues found:", "WARNING")
            for issue in issues:
                self._log_status(f"  - {issue}", "WARNING")
                
            # Attempt to install dependencies
            self._log_status("Attempting to install missing dependencies...")
            if self.debian_compat.install_dependencies():
                self._log_status("Dependencies installed successfully", "SUCCESS")
                # Re-check compatibility
                is_compatible, issues = self.debian_compat.check_system_compatibility()
            else:
                self._log_status("Failed to install dependencies", "ERROR")
                
        # Run safety validation
        self._log_status("Running comprehensive safety validation...")
        safety_report = self.safety_validator.run_full_safety_validation()
        
        # Save safety report
        report_path = self.safety_validator.save_safety_report(safety_report)
        if self.current_campaign:
            self.current_campaign.safety_reports.append(str(report_path))
            
        # Evaluate safety status
        if safety_report.overall_status == SafetyLevel.EMERGENCY:
            self._log_status("EMERGENCY: System not safe for testing!", "CRITICAL")
            return False, safety_report
        elif safety_report.overall_status == SafetyLevel.CRITICAL:
            self._log_status("CRITICAL: System not recommended for testing!", "ERROR")
            return False, safety_report
        elif safety_report.overall_status == SafetyLevel.WARNING:
            self._log_status("WARNING: System has issues but may proceed with caution", "WARNING")
        else:
            self._log_status("SAFE: System ready for testing", "SUCCESS")
            
        return True, safety_report
        
    def start_monitoring_systems(self) -> bool:
        """Start all monitoring systems"""
        
        self._log_status("Starting monitoring systems...")
        
        try:
            # Start thermal monitoring in the main testing framework
            # (This is handled by SMBIOSTokenTester)
            
            # Start comprehensive monitor if available
            monitor_script = self.work_dir / "monitoring" / "dsmil_comprehensive_monitor.py"
            if monitor_script.exists():
                self._log_status("Starting DSMIL comprehensive monitor...")
                proc = subprocess.Popen([
                    sys.executable, str(monitor_script)
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self.monitoring_processes.append(proc)
                
            # Start multi-terminal launcher if available (for manual monitoring)
            launcher_script = self.work_dir / "monitoring" / "multi_terminal_launcher.sh"
            if launcher_script.exists():
                self._log_status("Multi-terminal monitoring available")
                self._log_status(f"Run manually: {launcher_script}")
                
            self._log_status("Monitoring systems started", "SUCCESS")
            return True
            
        except Exception as e:
            self._log_status(f"Failed to start monitoring systems: {e}", "ERROR")
            return False
            
    def prepare_system_for_testing(self) -> bool:
        """Prepare system for token testing"""
        
        self._log_status("Preparing system for testing...")
        
        if self.current_campaign:
            self.current_campaign.phase = TestingPhase.SYSTEM_PREPARATION
            
        try:
            # Ensure DSMIL module is not loaded
            self._log_status("Checking DSMIL module status...")
            result = subprocess.run(['lsmod'], capture_output=True, text=True)
            
            if 'dsmil' in result.stdout:
                self._log_status("Unloading existing DSMIL module...")
                subprocess.run(['sudo', 'rmmod', 'dsmil-72dev'], 
                             capture_output=True, timeout=10)
                
            # Load Dell modules if available
            self._log_status("Loading Dell SMBIOS modules...")
            subprocess.run(['sudo', 'modprobe', 'dell-smbios'], 
                         capture_output=True)
            subprocess.run(['sudo', 'modprobe', 'dell-wmi'], 
                         capture_output=True)
            
            # Create backup if not exists
            self._log_status("Ensuring system backup exists...")
            backup_script = self.work_dir / "create-baseline-snapshot.sh"
            if backup_script.exists():
                # Check if recent backup exists
                recent_backups = list(self.work_dir.glob("baseline_*.tar.gz"))
                if not recent_backups:
                    self._log_status("Creating system backup...")
                    subprocess.run([str(backup_script)], timeout=300)
                    
            self._log_status("System preparation complete", "SUCCESS")
            return True
            
        except Exception as e:
            self._log_status(f"System preparation failed: {e}", "ERROR")
            return False
            
    def execute_token_testing(self, range_name: str) -> Tuple[bool, List[TokenTestResult]]:
        """Execute token testing for specified range"""
        
        self._log_status(f"Starting token testing for {range_name}...")
        
        if self.current_campaign:
            self.current_campaign.phase = TestingPhase.TOKEN_TESTING
            
        try:
            results = []
            
            if range_name == "single_0x0480":
                # Single token test
                self._log_status("Testing single token 0x0480...")
                result = self.tester.test_single_token(0x0480)
                results.append(result)
                
            elif range_name.startswith("Group_"):
                # Group testing
                self._log_status(f"Testing {range_name}...")
                group_results = self.tester.test_token_group(range_name)
                results.extend(group_results)
                
            elif range_name in self.tester.TARGET_RANGES:
                # Full range testing
                self._log_status(f"Testing full range {range_name}...")
                
                # Test each group in the range
                for group_name in self.tester.DSMIL_GROUPS:
                    if self.emergency_stop_triggered:
                        break
                        
                    self._log_status(f"Testing {group_name} in {range_name}...")
                    group_results = self.tester.test_token_group(group_name, delay_between_tests=10)
                    results.extend(group_results)
                    
                    # Inter-group delay
                    if not self.emergency_stop_triggered:
                        self._log_status("Inter-group delay (30 seconds)...")
                        time.sleep(30)
                        
            else:
                raise ValueError(f"Unknown range: {range_name}")
                
            # Update campaign statistics
            if self.current_campaign:
                self.current_campaign.total_tokens_tested += len(results)
                self.current_campaign.successful_tests += sum(1 for r in results if not r.errors)
                self.current_campaign.failed_tests += sum(1 for r in results if r.errors)
                
            success = not self.emergency_stop_triggered and len(results) > 0
            
            if success:
                self._log_status(f"Token testing for {range_name} completed successfully", "SUCCESS")
                if self.current_campaign:
                    self.current_campaign.completed_ranges.append(range_name)
            else:
                self._log_status(f"Token testing for {range_name} failed or aborted", "ERROR")
                if self.current_campaign:
                    self.current_campaign.failed_ranges.append(range_name)
                    
            return success, results
            
        except Exception as e:
            self._log_status(f"Token testing failed: {e}", "ERROR")
            if self.current_campaign:
                self.current_campaign.failed_ranges.append(range_name)
            return False, []
            
    def _emergency_stop(self):
        """Emergency stop all testing operations"""
        
        self._log_status("EMERGENCY STOP ACTIVATED", "CRITICAL")
        
        self.emergency_stop_triggered = True
        
        # Stop token tester
        if hasattr(self.tester, 'emergency_stop'):
            self.tester.emergency_stop()
            
        # Stop monitoring processes
        for proc in self.monitoring_processes:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except:
                try:
                    proc.kill()
                except:
                    pass
                    
        # Update campaign
        if self.current_campaign:
            self.current_campaign.emergency_stops += 1
            self.current_campaign.phase = TestingPhase.ABORTED
            
        # Run emergency stop script
        emergency_script = self.work_dir / "monitoring" / "emergency_stop.sh"
        if emergency_script.exists():
            try:
                subprocess.run([str(emergency_script)], timeout=30)
                self._log_status("Emergency stop script executed", "SUCCESS")
            except Exception as e:
                self._log_status(f"Emergency stop script failed: {e}", "ERROR")
                
    def cleanup_after_testing(self):
        """Cleanup after testing completion"""
        
        self._log_status("Starting cleanup after testing...")
        
        if self.current_campaign:
            self.current_campaign.phase = TestingPhase.CLEANUP
            
        try:
            # Stop thermal monitoring
            if hasattr(self.tester.thermal_monitor, 'stop_monitoring'):
                self.tester.thermal_monitor.stop_monitoring()
                
            # Stop monitoring processes
            for proc in self.monitoring_processes:
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except:
                    pass
                    
            # Unload DSMIL module if loaded
            result = subprocess.run(['lsmod'], capture_output=True, text=True)
            if 'dsmil' in result.stdout:
                subprocess.run(['sudo', 'rmmod', 'dsmil-72dev'], 
                             capture_output=True, timeout=10)
                             
            self._log_status("Cleanup completed", "SUCCESS")
            
        except Exception as e:
            self._log_status(f"Cleanup error: {e}", "ERROR")
            
    def generate_campaign_report(self) -> str:
        """Generate comprehensive campaign report"""
        
        if not self.current_campaign:
            return "No active campaign to report"
            
        campaign = self.current_campaign
        
        lines = []
        lines.append("=" * 80)
        lines.append("SMBIOS TOKEN TESTING CAMPAIGN REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        # Campaign summary
        lines.append(f"Campaign ID: {campaign.campaign_id}")
        lines.append(f"Start Time: {campaign.start_time}")
        lines.append(f"End Time: {campaign.end_time or 'In Progress'}")
        lines.append(f"Phase: {campaign.phase.value}")
        lines.append(f"Target Ranges: {', '.join(campaign.target_ranges)}")
        lines.append("")
        
        # Statistics
        lines.append("STATISTICS:")
        lines.append("-" * 40)
        lines.append(f"Total Tokens Tested: {campaign.total_tokens_tested}")
        lines.append(f"Successful Tests: {campaign.successful_tests}")
        lines.append(f"Failed Tests: {campaign.failed_tests}")
        lines.append(f"Emergency Stops: {campaign.emergency_stops}")
        lines.append(f"Completed Ranges: {len(campaign.completed_ranges)}")
        lines.append(f"Failed Ranges: {len(campaign.failed_ranges)}")
        lines.append("")
        
        # Range status
        if campaign.completed_ranges:
            lines.append("COMPLETED RANGES:")
            for range_name in campaign.completed_ranges:
                lines.append(f"  ✅ {range_name}")
            lines.append("")
            
        if campaign.failed_ranges:
            lines.append("FAILED RANGES:")
            for range_name in campaign.failed_ranges:
                lines.append(f"  ❌ {range_name}")
            lines.append("")
            
        # Status log (last 20 entries)
        if campaign.status_log:
            lines.append("RECENT STATUS LOG:")
            lines.append("-" * 40)
            for log_entry in campaign.status_log[-20:]:
                lines.append(log_entry)
            lines.append("")
            
        return "\n".join(lines)
        
    def save_campaign_report(self) -> Optional[Path]:
        """Save campaign report to file"""
        
        if not self.current_campaign:
            return None
            
        report_content = self.generate_campaign_report()
        
        filename = f"{self.current_campaign.campaign_id}_report.txt"
        report_path = self.testing_dir / filename
        
        with open(report_path, 'w') as f:
            f.write(report_content)
            
        return report_path
        
    def run_full_testing_campaign(self, scenario: str) -> bool:
        """Run complete testing campaign"""
        
        try:
            # Create campaign
            campaign = self.create_testing_campaign(scenario)
            
            # Phase 1: Pre-test validation
            self._log_status("=== PHASE 1: PRE-TEST VALIDATION ===")
            is_safe, safety_report = self.run_pre_test_validation()
            
            if not is_safe:
                self._log_status("Pre-test validation failed - aborting campaign", "CRITICAL")
                campaign.phase = TestingPhase.ABORTED
                return False
                
            # Phase 2: System preparation
            self._log_status("=== PHASE 2: SYSTEM PREPARATION ===")
            if not self.prepare_system_for_testing():
                self._log_status("System preparation failed - aborting campaign", "CRITICAL")
                campaign.phase = TestingPhase.ABORTED
                return False
                
            # Start monitoring
            self.start_monitoring_systems()
            
            # Phase 3: Token testing
            self._log_status("=== PHASE 3: TOKEN TESTING ===")
            all_successful = True
            
            for range_name in campaign.target_ranges:
                if self.emergency_stop_triggered:
                    break
                    
                success, results = self.execute_token_testing(range_name)
                if not success:
                    all_successful = False
                    
                # Save intermediate results
                if hasattr(self.tester, 'save_test_results'):
                    self.tester.save_test_results()
                    
            # Phase 4: Cleanup
            self._log_status("=== PHASE 4: CLEANUP ===")
            self.cleanup_after_testing()
            
            # Finalize campaign
            campaign.end_time = datetime.now(timezone.utc)
            campaign.phase = TestingPhase.COMPLETED if all_successful else TestingPhase.ABORTED
            
            # Save final report
            report_path = self.save_campaign_report()
            if report_path:
                self._log_status(f"Campaign report saved: {report_path}")
                
            return all_successful
            
        except Exception as e:
            self._log_status(f"Campaign failed with error: {e}", "CRITICAL")
            self._emergency_stop()
            return False

def main():
    """Main orchestration interface"""
    
    print("🎯 SMBIOS Token Testing Orchestration System v1.0.0")
    print("Dell Latitude 5450 MIL-SPEC - TESTBED Agent")
    print("=" * 70)
    
    orchestrator = TokenTestingOrchestrator()
    
    # Display available scenarios
    print("\nAvailable Testing Scenarios:")
    print("-" * 40)
    
    for key, scenario in orchestrator.testing_scenarios.items():
        print(f"{key:15} - {scenario['name']}")
        print(f"{'':15}   {scenario['description']}")
        print(f"{'':15}   Estimated time: {scenario['estimated_time']}")
        print()
        
    # Get user selection
    print("Select testing scenario:")
    for i, (key, scenario) in enumerate(orchestrator.testing_scenarios.items(), 1):
        print(f"{i}. {scenario['name']}")
        
    try:
        choice = input("\nEnter choice (1-4): ").strip()
        choice_num = int(choice)
        
        if 1 <= choice_num <= len(orchestrator.testing_scenarios):
            scenario_keys = list(orchestrator.testing_scenarios.keys())
            selected_scenario = scenario_keys[choice_num - 1]
            scenario_info = orchestrator.testing_scenarios[selected_scenario]
            
            print(f"\nSelected: {scenario_info['name']}")
            print(f"Estimated time: {scenario_info['estimated_time']}")
            print(f"Ranges: {', '.join(scenario_info['ranges'])}")
            
            confirm = input("\nProceed with this scenario? (y/N): ").strip().lower()
            
            if confirm == 'y':
                print(f"\n🚀 Starting testing campaign: {scenario_info['name']}")
                print("=" * 70)
                
                # Run the campaign
                success = orchestrator.run_full_testing_campaign(selected_scenario)
                
                if success:
                    print("\n✅ Testing campaign completed successfully!")
                else:
                    print("\n❌ Testing campaign failed or was aborted")
                    
                # Show final report
                if orchestrator.current_campaign:
                    print("\nFINAL REPORT:")
                    print("=" * 40)
                    report = orchestrator.generate_campaign_report()
                    print(report)
                    
            else:
                print("Testing cancelled by user")
                
        else:
            print("Invalid choice")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Testing interrupted by user")
        orchestrator._emergency_stop()
        return 1
        
    except ValueError:
        print("Invalid input - please enter a number")
        return 1
        
    except Exception as e:
        print(f"\n❌ Orchestration error: {e}")
        orchestrator._emergency_stop()
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())