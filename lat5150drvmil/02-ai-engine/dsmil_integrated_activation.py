#!/usr/bin/env python3
"""
DSMIL Integrated Activation System
===================================
End-to-end workflow combining:
- ML-enhanced hardware discovery
- Intelligent device activation
- Real-time monitoring and safety checks

This is the MISSION-CRITICAL integration layer for smooth device activation.

Author: LAT5150DRVMIL AI Platform
Classification: DSMIL Integrated System
"""

import os
import sys
import json
import logging
import time
import curses
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Add script directory to path
script_dir = Path(__file__).parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

# Import our modules
try:
    from dsmil_ml_discovery import DSMILMLDiscovery, HardwareDevice, DeviceSafetyLevel
    from dsmil_device_activation import DSMILDeviceActivator, ActivationResult, ActivationStatus
    from dsmil_subsystem_controller import DSMILSubsystemController
    ML_AVAILABLE = True
except ImportError as e:
    logging.warning(f"ML discovery not available: {e}")
    ML_AVAILABLE = False

# Configure logging
logging.basicConfig(
    filename='/tmp/dsmil_integrated_activation.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WorkflowStage(Enum):
    """Activation workflow stages"""
    INITIALIZING = "initializing"
    DISCOVERING = "discovering"
    ANALYZING = "analyzing"
    ACTIVATING = "activating"
    MONITORING = "monitoring"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class WorkflowStatus:
    """Workflow execution status"""
    stage: WorkflowStage
    current_device: Optional[int]
    devices_discovered: int
    devices_activated: int
    devices_failed: int
    thermal_status: str
    message: str


class DSMILIntegratedActivation:
    """Integrated activation system with ML discovery"""

    def __init__(self):
        """Initialize integrated system"""
        self.workflow_status = WorkflowStatus(
            stage=WorkflowStage.INITIALIZING,
            current_device=None,
            devices_discovered=0,
            devices_activated=0,
            devices_failed=0,
            thermal_status="unknown",
            message="Initializing..."
        )

        self.discovered_devices: Dict[int, HardwareDevice] = {}
        self.activation_results: List[ActivationResult] = []
        self.activation_sequence: List[int] = []

        # Initialize components
        try:
            self.discovery = DSMILMLDiscovery() if ML_AVAILABLE else None
            self.activator = DSMILDeviceActivator() if ML_AVAILABLE else None
            self.controller = DSMILSubsystemController() if ML_AVAILABLE else None
            self.components_available = True
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            self.discovery = None
            self.activator = None
            self.controller = None
            self.components_available = False

        logger.info("Integrated activation system initialized")

    def run_discovery_phase(self, progress_callback=None) -> bool:
        """Phase 1: Discover all hardware devices"""
        logger.info("=" * 70)
        logger.info("PHASE 1: ML-ENHANCED HARDWARE DISCOVERY")
        logger.info("=" * 70)

        self.workflow_status.stage = WorkflowStage.DISCOVERING
        self.workflow_status.message = "Scanning hardware interfaces..."

        if not self.components_available or not self.discovery:
            logger.error("Discovery components not available")
            self.workflow_status.stage = WorkflowStage.FAILED
            self.workflow_status.message = "Discovery system unavailable"
            return False

        try:
            # Run comprehensive discovery
            if progress_callback:
                progress_callback("Scanning SMBIOS tokens...")

            self.discovered_devices = self.discovery.discover_all_devices()

            self.workflow_status.devices_discovered = len(self.discovered_devices)
            self.workflow_status.message = f"Discovered {len(self.discovered_devices)} devices"

            logger.info(f"Discovery complete: {len(self.discovered_devices)} devices found")
            return True

        except Exception as e:
            logger.error(f"Discovery phase failed: {e}")
            self.workflow_status.stage = WorkflowStage.FAILED
            self.workflow_status.message = f"Discovery error: {e}"
            return False

    def run_analysis_phase(self, progress_callback=None) -> bool:
        """Phase 2: Analyze devices and plan activation"""
        logger.info("=" * 70)
        logger.info("PHASE 2: ML ANALYSIS AND ACTIVATION PLANNING")
        logger.info("=" * 70)

        self.workflow_status.stage = WorkflowStage.ANALYZING
        self.workflow_status.message = "Analyzing devices with ML..."

        if not self.discovery:
            return False

        try:
            # Generate optimal activation sequence
            if progress_callback:
                progress_callback("Calculating activation sequence...")

            self.activation_sequence = self.discovery.get_activation_sequence()

            # Log analysis results
            safe_count = sum(1 for d in self.discovered_devices.values()
                           if d.safety_level == DeviceSafetyLevel.SAFE)
            monitored_count = sum(1 for d in self.discovered_devices.values()
                                if d.safety_level == DeviceSafetyLevel.MONITORED)
            quarantined_count = sum(1 for d in self.discovered_devices.values()
                                   if d.safety_level == DeviceSafetyLevel.QUARANTINED)

            logger.info(f"Analysis complete:")
            logger.info(f"  Safe devices: {safe_count}")
            logger.info(f"  Monitored devices: {monitored_count}")
            logger.info(f"  Quarantined devices: {quarantined_count}")
            logger.info(f"  Activation sequence: {len(self.activation_sequence)} devices")

            self.workflow_status.message = f"Ready to activate {len(self.activation_sequence)} devices"
            return True

        except Exception as e:
            logger.error(f"Analysis phase failed: {e}")
            self.workflow_status.stage = WorkflowStage.FAILED
            self.workflow_status.message = f"Analysis error: {e}"
            return False

    def run_activation_phase(self, progress_callback=None, interactive=False) -> bool:
        """Phase 3: Activate devices in optimal sequence"""
        logger.info("=" * 70)
        logger.info("PHASE 3: INTELLIGENT DEVICE ACTIVATION")
        logger.info("=" * 70)

        self.workflow_status.stage = WorkflowStage.ACTIVATING
        self.workflow_status.message = "Activating devices..."

        if not self.activator or not len(self.activation_sequence):
            logger.error("Activation prerequisites not met")
            return False

        try:
            total_devices = len(self.activation_sequence)

            for idx, device_id in enumerate(self.activation_sequence):
                device = self.discovered_devices.get(device_id)
                if not device:
                    continue

                self.workflow_status.current_device = device_id
                self.workflow_status.message = f"Activating {device.name} ({idx+1}/{total_devices})..."

                if progress_callback:
                    progress_callback(self.workflow_status.message)

                # Interactive confirmation for monitored devices
                if interactive and device.safety_level == DeviceSafetyLevel.MONITORED:
                    if progress_callback:
                        response = progress_callback(
                            f"Activate monitored device {device.name}? (y/n)",
                            require_input=True
                        )
                        if response.lower() != 'y':
                            logger.info(f"Skipping {device.name} (user declined)")
                            continue

                # Activate device
                result = self.activator.activate_device(device_id)
                self.activation_results.append(result)

                if result.success:
                    self.workflow_status.devices_activated += 1
                    logger.info(f"✓ {device.name} activated successfully")
                else:
                    self.workflow_status.devices_failed += 1
                    logger.warning(f"✗ {device.name} activation failed: {result.message}")

                # Check thermal conditions
                if self.controller:
                    thermal = self.controller.get_thermal_status_enhanced()
                    self.workflow_status.thermal_status = thermal.get('overall_status', 'unknown')

                    if thermal.get('overall_status') == 'critical':
                        logger.error("THERMAL CRITICAL - Halting activation sequence")
                        self.workflow_status.stage = WorkflowStage.FAILED
                        self.workflow_status.message = "Halted due to thermal critical"
                        return False

                # Brief pause between activations
                time.sleep(0.5)

            logger.info(f"Activation phase complete: {self.workflow_status.devices_activated}/{total_devices} successful")
            return True

        except Exception as e:
            logger.error(f"Activation phase failed: {e}")
            self.workflow_status.stage = WorkflowStage.FAILED
            self.workflow_status.message = f"Activation error: {e}"
            return False

    def run_monitoring_phase(self, duration: int = 30) -> bool:
        """Phase 4: Monitor activated devices"""
        logger.info("=" * 70)
        logger.info("PHASE 4: POST-ACTIVATION MONITORING")
        logger.info("=" * 70)

        self.workflow_status.stage = WorkflowStage.MONITORING
        self.workflow_status.message = f"Monitoring for {duration}s..."

        if not self.controller:
            logger.warning("Monitoring unavailable")
            return True

        try:
            start_time = time.time()
            check_interval = 5

            while time.time() - start_time < duration:
                # Check thermal status
                thermal = self.controller.get_thermal_status_enhanced()
                max_temp = thermal.get('max_temp', 0)

                logger.info(f"Monitoring: Temp={max_temp}°C, Status={thermal.get('overall_status')}")

                if thermal.get('overall_status') == 'critical':
                    logger.error("CRITICAL: Thermal emergency detected")
                    self.workflow_status.message = "THERMAL CRITICAL"
                    return False

                time.sleep(check_interval)

            logger.info("Monitoring phase complete - all systems stable")
            return True

        except Exception as e:
            logger.error(f"Monitoring phase failed: {e}")
            return False

    def run_full_workflow(self, progress_callback=None, interactive=False, monitor_duration=30) -> bool:
        """Execute complete end-to-end activation workflow"""
        logger.info("=" * 80)
        logger.info("DSMIL INTEGRATED ACTIVATION WORKFLOW - STARTING")
        logger.info("=" * 80)

        start_time = time.time()

        # Phase 1: Discovery
        if not self.run_discovery_phase(progress_callback):
            logger.error("Workflow failed at discovery phase")
            return False

        # Phase 2: Analysis
        if not self.run_analysis_phase(progress_callback):
            logger.error("Workflow failed at analysis phase")
            return False

        # Phase 3: Activation
        if not self.run_activation_phase(progress_callback, interactive):
            logger.error("Workflow failed at activation phase")
            return False

        # Phase 4: Monitoring
        if not self.run_monitoring_phase(monitor_duration):
            logger.error("Workflow failed at monitoring phase")
            return False

        # Complete
        self.workflow_status.stage = WorkflowStage.COMPLETE
        elapsed_time = time.time() - start_time
        self.workflow_status.message = f"Complete in {elapsed_time:.1f}s"

        logger.info("=" * 80)
        logger.info(f"WORKFLOW COMPLETE - {self.workflow_status.devices_activated} devices activated")
        logger.info(f"Elapsed time: {elapsed_time:.1f}s")
        logger.info("=" * 80)

        return True

    def export_workflow_report(self, output_path: Optional[Path] = None) -> Dict:
        """Export comprehensive workflow report"""
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'workflow_status': {
                'stage': self.workflow_status.stage.value,
                'message': self.workflow_status.message,
                'devices_discovered': self.workflow_status.devices_discovered,
                'devices_activated': self.workflow_status.devices_activated,
                'devices_failed': self.workflow_status.devices_failed,
                'thermal_status': self.workflow_status.thermal_status,
            },
            'discovered_devices': len(self.discovered_devices),
            'activation_sequence': [f"0x{d:04X}" for d in self.activation_sequence],
            'activation_results': [
                {
                    'device_id': f"0x{r.device_id:04X}",
                    'device_name': r.device_name,
                    'success': r.success,
                    'method': r.method.value if r.method else None,
                    'message': r.message,
                    'thermal_impact': r.thermal_impact,
                }
                for r in self.activation_results
            ]
        }

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Workflow report saved to {output_path}")

        return report


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="DSMIL Integrated Activation System")
    parser.add_argument('--interactive', action='store_true',
                       help="Interactive mode (confirm each device)")
    parser.add_argument('--monitor-duration', type=int, default=30,
                       help="Post-activation monitoring duration (seconds)")
    parser.add_argument('--report', type=str,
                       help="Save workflow report to file")
    parser.add_argument('--no-activation', action='store_true',
                       help="Discovery and analysis only (no activation)")

    args = parser.parse_args()

    print("=" * 70)
    print("DSMIL INTEGRATED ACTIVATION SYSTEM")
    print("End-to-End ML-Enhanced Hardware Discovery and Activation")
    print("=" * 70)
    print()

    if os.geteuid() != 0:
        print("⚠️  WARNING: Not running as root.")
        print("   Some operations may fail without root privileges.")
        print("   Run with: sudo python3 dsmil_integrated_activation.py")
        print()
        input("Press ENTER to continue anyway...")

    # Initialize system
    system = DSMILIntegratedActivation()

    if not system.components_available:
        print("ERROR: Required components not available.")
        print("Ensure DSMIL kernel driver is built and dependencies are installed.")
        return 1

    # Progress callback for CLI
    def progress_callback(message, require_input=False):
        print(f"  {message}")
        if require_input:
            return input("    > ")
        return None

    # Run workflow
    if args.no_activation:
        # Discovery and analysis only
        success = system.run_discovery_phase(progress_callback)
        if success:
            success = system.run_analysis_phase(progress_callback)
    else:
        # Full workflow
        success = system.run_full_workflow(
            progress_callback=progress_callback,
            interactive=args.interactive,
            monitor_duration=args.monitor_duration
        )

    # Export report
    report_path = Path(args.report) if args.report else Path('/tmp/dsmil_integrated_workflow_report.json')
    report = system.export_workflow_report(report_path)

    # Print summary
    print()
    print("=" * 70)
    print("WORKFLOW SUMMARY")
    print("=" * 70)
    print(f"Stage: {report['workflow_status']['stage'].upper()}")
    print(f"Devices Discovered: {report['workflow_status']['devices_discovered']}")
    print(f"Devices Activated: {report['workflow_status']['devices_activated']}")
    print(f"Devices Failed: {report['workflow_status']['devices_failed']}")
    print(f"Thermal Status: {report['workflow_status']['thermal_status']}")
    print(f"Report saved to: {report_path}")
    print("=" * 70)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
