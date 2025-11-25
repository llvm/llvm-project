#!/usr/bin/env python3
"""
Unified Debug Orchestrator for DSMIL Systems

Orchestrates all debugging components and provides unified interface for
comprehensive DSMIL system analysis and debugging.

Dell Latitude 5450 MIL-SPEC - Master Debug Controller
"""

import os
import sys
import time
import signal
import threading
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Any
import logging
import json
import argparse

# Import our debugging modules
try:
    from dsmil_debug_infrastructure import DSMILDebugger
    from kernel_trace_analyzer import KernelTraceAnalyzer
    from smbios_correlation_engine import SMBIOSCorrelationEngine
    from memory_pattern_analyzer import MemoryPatternAnalyzer
except ImportError as e:
    print(f"Error importing debugging modules: {e}")
    print("Ensure all debugging modules are in the same directory")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedDebugOrchestrator:
    """Master orchestrator for all DSMIL debugging components"""
    
    def __init__(self, base_debug_dir: str = "/tmp/dsmil_unified_debug"):
        self.base_debug_dir = Path(base_debug_dir)
        self.base_debug_dir.mkdir(exist_ok=True)
        
        # Initialize component directories
        self.component_dirs = {
            'infrastructure': str(self.base_debug_dir / "infrastructure"),
            'kernel_trace': str(self.base_debug_dir / "kernel_trace"),
            'correlation': str(self.base_debug_dir / "correlation"),
            'memory_analysis': str(self.base_debug_dir / "memory_analysis")
        }
        
        # Create component directories
        for comp_dir in self.component_dirs.values():
            Path(comp_dir).mkdir(exist_ok=True)
        
        # Initialize debugging components
        self.debugger = DSMILDebugger(self.component_dirs['infrastructure'])
        self.kernel_tracer = KernelTraceAnalyzer(self.component_dirs['kernel_trace'])
        self.correlation_engine = SMBIOSCorrelationEngine(
            str(self.base_debug_dir / "correlation.db")
        )
        self.memory_analyzer = MemoryPatternAnalyzer(self.component_dirs['memory_analysis'])
        
        # Component states
        self.components = {
            'debugger': {'instance': self.debugger, 'running': False},
            'kernel_tracer': {'instance': self.kernel_tracer, 'running': False},
            'correlation_engine': {'instance': self.correlation_engine, 'running': False},
            'memory_analyzer': {'instance': self.memory_analyzer, 'running': False}
        }
        
        # Orchestrator state
        self.running = False
        self.start_time = None
        self.orchestrator_thread = None
        
        # Session configuration
        self.session_config = {
            'duration_seconds': 300,  # Default 5 minutes
            'auto_report_interval': 60,  # Generate reports every minute
            'component_sync_interval': 5,  # Sync components every 5 seconds
            'enable_cross_correlation': True,
            'enable_realtime_alerts': True,
            'alert_thresholds': {
                'error_rate': 0.1,  # 10% error rate triggers alert
                'memory_anomaly_count': 5,
                'correlation_confidence': 0.8
            }
        }
        
        # Cross-component data sharing
        self.shared_data = {
            'token_operations': [],
            'system_events': [],
            'correlations': [],
            'alerts': []
        }
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("Unified Debug Orchestrator initialized")

    def configure_session(self, **kwargs):
        """Configure debugging session parameters"""
        for key, value in kwargs.items():
            if key in self.session_config:
                self.session_config[key] = value
                logger.info(f"Session config updated: {key} = {value}")

    def start_unified_debugging(self):
        """Start all debugging components in coordinated manner"""
        if self.running:
            logger.warning("Debugging session already running")
            return
        
        self.running = True
        self.start_time = datetime.now()
        
        logger.info("Starting unified debugging session")
        logger.info(f"Session duration: {self.session_config['duration_seconds']} seconds")
        logger.info(f"Output directory: {self.base_debug_dir}")
        
        # Start all components
        try:
            # Start infrastructure debugger
            self.debugger.start_monitoring()
            self.components['debugger']['running'] = True
            logger.info("✓ Infrastructure debugger started")
            
            # Start kernel tracer (no start method, will be controlled manually)
            self.components['kernel_tracer']['running'] = True
            logger.info("✓ Kernel tracer initialized")
            
            # Start correlation engine
            self.correlation_engine.start_monitoring()
            self.components['correlation_engine']['running'] = True
            logger.info("✓ Correlation engine started")
            
            # Start memory analyzer
            self.memory_analyzer.start_monitoring()
            self.components['memory_analyzer']['running'] = True
            logger.info("✓ Memory pattern analyzer started")
            
            # Start orchestrator coordination thread
            self.orchestrator_thread = threading.Thread(
                target=self._orchestration_loop,
                daemon=True
            )
            self.orchestrator_thread.start()
            logger.info("✓ Orchestration thread started")
            
            logger.info("All debugging components started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start debugging components: {e}")
            self.stop_unified_debugging()
            raise

    def stop_unified_debugging(self):
        """Stop all debugging components and generate final report"""
        if not self.running:
            return
        
        logger.info("Stopping unified debugging session")
        self.running = False
        
        # Stop all components
        try:
            if self.components['debugger']['running']:
                self.debugger.stop_monitoring()
                self.components['debugger']['running'] = False
                logger.info("✓ Infrastructure debugger stopped")
            
            if self.components['correlation_engine']['running']:
                self.correlation_engine.stop_monitoring()
                self.components['correlation_engine']['running'] = False
                logger.info("✓ Correlation engine stopped")
            
            if self.components['memory_analyzer']['running']:
                self.memory_analyzer.stop_monitoring()
                self.components['memory_analyzer']['running'] = False
                logger.info("✓ Memory pattern analyzer stopped")
            
            # Wait for orchestrator thread
            if self.orchestrator_thread and self.orchestrator_thread.is_alive():
                self.orchestrator_thread.join(timeout=5)
                logger.info("✓ Orchestration thread stopped")
            
        except Exception as e:
            logger.error(f"Error stopping components: {e}")
        
        # Generate final unified report
        try:
            final_report = self.generate_unified_report()
            logger.info(f"Final unified report: {final_report}")
        except Exception as e:
            logger.error(f"Failed to generate final report: {e}")
        
        logger.info("Unified debugging session stopped")

    def _orchestration_loop(self):
        """Main orchestration loop for cross-component coordination"""
        logger.info("Orchestration loop started")
        
        last_report_time = datetime.now()
        last_sync_time = datetime.now()
        
        try:
            while self.running:
                current_time = datetime.now()
                
                # Check session duration
                if (current_time - self.start_time).total_seconds() > self.session_config['duration_seconds']:
                    logger.info("Session duration reached, stopping debugging")
                    self.stop_unified_debugging()
                    break
                
                # Periodic component synchronization
                if (current_time - last_sync_time).total_seconds() >= self.session_config['component_sync_interval']:
                    self._synchronize_components()
                    last_sync_time = current_time
                
                # Periodic report generation
                if (current_time - last_report_time).total_seconds() >= self.session_config['auto_report_interval']:
                    self._generate_intermediate_report()
                    last_report_time = current_time
                
                # Cross-correlation analysis
                if self.session_config['enable_cross_correlation']:
                    self._perform_cross_correlation()
                
                # Real-time alert checking
                if self.session_config['enable_realtime_alerts']:
                    self._check_realtime_alerts()
                
                time.sleep(1)  # Main orchestration loop runs every second
                
        except Exception as e:
            logger.error(f"Orchestration loop error: {e}")
        finally:
            logger.info("Orchestration loop ended")

    def _synchronize_components(self):
        """Synchronize data between debugging components"""
        try:
            # Extract data from infrastructure debugger
            if hasattr(self.debugger, 'tokens'):
                token_data = []
                for token_id, token_state in self.debugger.tokens.items():
                    if token_state.access_count > 0:
                        token_data.append({
                            'token_id': token_id,
                            'access_count': token_state.access_count,
                            'last_access': token_state.last_access,
                            'error_count': token_state.error_count
                        })
                
                # Share with correlation engine
                for token_info in token_data:
                    if token_info['last_access']:
                        self.correlation_engine.record_token_operation(
                            token_id=token_info['token_id'],
                            operation='sync_update',
                            duration_ms=None
                        )
            
            # Extract system events from kernel tracer
            if hasattr(self.kernel_tracer, 'pattern_matches'):
                for pattern_name, matches in self.kernel_tracer.pattern_matches.items():
                    for match in matches[-5:]:  # Last 5 matches
                        self.correlation_engine.record_system_event(
                            event_type=pattern_name,
                            source='kernel_tracer',
                            message=match['message'][:200],
                            severity='info'
                        )
            
            logger.debug("Component synchronization completed")
            
        except Exception as e:
            logger.error(f"Component synchronization error: {e}")

    def _perform_cross_correlation(self):
        """Perform cross-component correlation analysis"""
        try:
            current_time = datetime.now()
            
            # Get recent data from all components
            recent_cutoff = current_time - timedelta(seconds=30)
            
            # Correlate token operations with memory patterns
            memory_patterns = [
                p for p in self.memory_analyzer.detected_patterns
                if p.timestamp > recent_cutoff
            ]
            
            token_operations = [
                op for op in self.correlation_engine.token_operations
                if op.timestamp > recent_cutoff
            ]
            
            # Look for correlations
            for token_op in token_operations:
                for pattern in memory_patterns:
                    time_diff = abs((pattern.timestamp - token_op.timestamp).total_seconds())
                    if time_diff <= 10.0:  # Within 10 seconds
                        correlation = {
                            'timestamp': current_time,
                            'type': 'token_memory_correlation',
                            'token_id': token_op.token_id,
                            'memory_pattern': pattern.pattern_type,
                            'time_diff': time_diff,
                            'confidence': 0.8 if time_diff < 5.0 else 0.6
                        }
                        
                        self.shared_data['correlations'].append(correlation)
                        
                        if correlation['confidence'] > 0.7:
                            logger.info(f"Cross-correlation detected: Token 0x{token_op.token_id:04X} "
                                      f"↔ {pattern.pattern_type} (confidence: {correlation['confidence']:.2f})")
            
        except Exception as e:
            logger.error(f"Cross-correlation analysis error: {e}")

    def _check_realtime_alerts(self):
        """Check for conditions that warrant real-time alerts"""
        try:
            current_time = datetime.now()
            recent_cutoff = current_time - timedelta(minutes=1)
            
            # Check error rate
            recent_errors = len([
                e for e in self.shared_data['system_events']
                if e.get('event_type') == 'error' and e.get('timestamp', datetime.min) > recent_cutoff
            ])
            
            total_recent_events = len([
                e for e in self.shared_data['system_events']
                if e.get('timestamp', datetime.min) > recent_cutoff
            ])
            
            if total_recent_events > 0:
                error_rate = recent_errors / total_recent_events
                if error_rate > self.session_config['alert_thresholds']['error_rate']:
                    alert = {
                        'timestamp': current_time,
                        'type': 'high_error_rate',
                        'value': error_rate,
                        'threshold': self.session_config['alert_thresholds']['error_rate'],
                        'message': f"High error rate detected: {error_rate:.2%}"
                    }
                    
                    self.shared_data['alerts'].append(alert)
                    logger.warning(f"ALERT: {alert['message']}")
            
            # Check memory anomalies
            recent_memory_anomalies = len([
                p for p in self.memory_analyzer.detected_patterns
                if 'anomaly' in p.pattern_type and p.timestamp > recent_cutoff
            ])
            
            if recent_memory_anomalies > self.session_config['alert_thresholds']['memory_anomaly_count']:
                alert = {
                    'timestamp': current_time,
                    'type': 'memory_anomalies',
                    'value': recent_memory_anomalies,
                    'threshold': self.session_config['alert_thresholds']['memory_anomaly_count'],
                    'message': f"Multiple memory anomalies detected: {recent_memory_anomalies}"
                }
                
                self.shared_data['alerts'].append(alert)
                logger.warning(f"ALERT: {alert['message']}")
            
        except Exception as e:
            logger.error(f"Real-time alert checking error: {e}")

    def _generate_intermediate_report(self):
        """Generate intermediate report during debugging session"""
        try:
            timestamp = int(datetime.now().timestamp())
            report_file = self.base_debug_dir / f"intermediate_report_{timestamp}.json"
            
            # Collect status from all components
            component_status = {}
            for name, comp_info in self.components.items():
                component_status[name] = {
                    'running': comp_info['running'],
                    'status': 'active' if comp_info['running'] else 'stopped'
                }
                
                # Add component-specific stats
                if name == 'debugger' and hasattr(self.debugger, 'stats'):
                    component_status[name]['stats'] = dict(self.debugger.stats)
                elif name == 'correlation_engine' and hasattr(self.correlation_engine, 'token_operations'):
                    component_status[name]['stats'] = {
                        'token_operations': len(self.correlation_engine.token_operations),
                        'system_events': len(self.correlation_engine.system_events),
                        'correlations': len(self.correlation_engine.correlations)
                    }
                elif name == 'memory_analyzer':
                    component_status[name]['stats'] = {
                        'memory_accesses': len(self.memory_analyzer.memory_accesses),
                        'detected_patterns': len(self.memory_analyzer.detected_patterns)
                    }
            
            elapsed_time = (datetime.now() - self.start_time).total_seconds()
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'session_elapsed_seconds': elapsed_time,
                'session_progress': elapsed_time / self.session_config['duration_seconds'],
                'component_status': component_status,
                'shared_data_summary': {
                    'correlations': len(self.shared_data['correlations']),
                    'alerts': len(self.shared_data['alerts'])
                },
                'recent_alerts': self.shared_data['alerts'][-5:],  # Last 5 alerts
                'system_health': {
                    'all_components_running': all(c['running'] for c in self.components.values()),
                    'active_components': sum(1 for c in self.components.values() if c['running']),
                    'total_components': len(self.components)
                }
            }
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.debug(f"Intermediate report generated: {report_file}")
            
        except Exception as e:
            logger.error(f"Intermediate report generation error: {e}")

    def execute_token_test_sequence(self, token_range: range, operations: List[str] = None):
        """Execute coordinated token testing across all components"""
        if operations is None:
            operations = ['read']
        
        logger.info(f"Starting coordinated token test: {len(token_range)} tokens, "
                   f"{len(operations)} operations each")
        
        results = []
        
        for token_id in token_range:
            for operation in operations:
                try:
                    # Record test start
                    start_time = datetime.now()
                    
                    # Execute through infrastructure debugger
                    debug_result = self.debugger.test_token_response(token_id, operation)
                    
                    # Record in correlation engine
                    self.correlation_engine.record_token_operation(
                        token_id=token_id,
                        operation=operation,
                        duration_ms=(datetime.now() - start_time).total_seconds() * 1000
                    )
                    
                    # Record in memory analyzer (simulate memory access)
                    if 0x0480 <= token_id <= 0x04C7:
                        group_id = (token_id - 0x0480) // 12
                        device_id = (token_id - 0x0480) % 12
                        # Simulate device register access
                        register_addr = 0x52000000 + (group_id * 0x10000) + (device_id * 0x1000)
                        
                        self.memory_analyzer.record_memory_access(
                            address=register_addr,
                            size=4,
                            operation=f'token_{operation}',
                            context=f'token_0x{token_id:04X}_{operation}',
                            caller='unified_orchestrator'
                        )
                    
                    result = {
                        'token_id': f'0x{token_id:04X}',
                        'operation': operation,
                        'timestamp': start_time.isoformat(),
                        'success': 'error' not in debug_result.get('test_result', {}),
                        'debug_result': debug_result
                    }
                    
                    results.append(result)
                    
                    # Small delay between operations
                    time.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"Token test error 0x{token_id:04X} {operation}: {e}")
                    results.append({
                        'token_id': f'0x{token_id:04X}',
                        'operation': operation,
                        'error': str(e),
                        'success': False
                    })
        
        # Save test results
        test_file = self.base_debug_dir / f"token_test_sequence_{int(datetime.now().timestamp())}.json"
        with open(test_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Token test sequence completed: {test_file}")
        return results

    def generate_unified_report(self) -> str:
        """Generate comprehensive unified debug report"""
        timestamp = int(datetime.now().timestamp())
        report_file = self.base_debug_dir / f"unified_debug_report_{timestamp}.json"
        
        try:
            # Collect reports from all components
            component_reports = {}
            
            # Infrastructure debugger report
            try:
                infra_report_file = self.debugger.generate_debug_report()
                with open(infra_report_file, 'r') as f:
                    component_reports['infrastructure'] = json.load(f)
            except Exception as e:
                logger.error(f"Failed to get infrastructure report: {e}")
                component_reports['infrastructure'] = {'error': str(e)}
            
            # Kernel tracer report
            try:
                kernel_report_file = self.kernel_tracer.generate_trace_report()
                with open(kernel_report_file, 'r') as f:
                    component_reports['kernel_trace'] = json.load(f)
            except Exception as e:
                logger.error(f"Failed to get kernel trace report: {e}")
                component_reports['kernel_trace'] = {'error': str(e)}
            
            # Correlation engine report
            try:
                corr_report_file = self.correlation_engine.generate_correlation_report()
                with open(corr_report_file, 'r') as f:
                    component_reports['correlation'] = json.load(f)
            except Exception as e:
                logger.error(f"Failed to get correlation report: {e}")
                component_reports['correlation'] = {'error': str(e)}
            
            # Memory analyzer report
            try:
                memory_report_file = self.memory_analyzer.generate_memory_report()
                with open(memory_report_file, 'r') as f:
                    component_reports['memory_analysis'] = json.load(f)
            except Exception as e:
                logger.error(f"Failed to get memory analysis report: {e}")
                component_reports['memory_analysis'] = {'error': str(e)}
            
            # Session summary
            session_duration = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
            
            unified_report = {
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'session_start': self.start_time.isoformat() if self.start_time else None,
                    'session_duration_seconds': session_duration,
                    'orchestrator_version': '1.0.0',
                    'target_system': 'Dell Latitude 5450 MIL-SPEC',
                    'dsmil_devices': '72 devices (6 groups × 12 devices)'
                },
                'session_configuration': self.session_config,
                'component_reports': component_reports,
                'cross_component_analysis': {
                    'correlations_found': len(self.shared_data.get('correlations', [])),
                    'alerts_triggered': len(self.shared_data.get('alerts', [])),
                    'synchronization_events': 'tracked_internally'
                },
                'unified_findings': self._generate_unified_findings(),
                'recommendations': self._generate_unified_recommendations(),
                'debug_artifacts': {
                    'base_directory': str(self.base_debug_dir),
                    'component_directories': self.component_dirs,
                    'database_files': ['correlation.db'],
                    'report_files': 'auto_generated'
                }
            }
            
            with open(report_file, 'w') as f:
                json.dump(unified_report, f, indent=2, default=str)
            
            logger.info(f"Unified debug report generated: {report_file}")
            return str(report_file)
            
        except Exception as e:
            logger.error(f"Unified report generation error: {e}")
            # Generate minimal error report
            error_report = {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'session_duration': session_duration if 'session_duration' in locals() else 0
            }
            
            with open(report_file, 'w') as f:
                json.dump(error_report, f, indent=2)
            
            return str(report_file)

    def _generate_unified_findings(self) -> Dict[str, Any]:
        """Generate unified analysis findings"""
        findings = {
            'system_behavior': {},
            'token_patterns': {},
            'memory_patterns': {},
            'error_analysis': {},
            'performance_metrics': {}
        }
        
        try:
            # Analyze cross-component correlations
            correlations = self.shared_data.get('correlations', [])
            if correlations:
                findings['system_behavior']['correlation_strength'] = {
                    'high_confidence': len([c for c in correlations if c.get('confidence', 0) > 0.8]),
                    'medium_confidence': len([c for c in correlations if 0.5 < c.get('confidence', 0) <= 0.8]),
                    'low_confidence': len([c for c in correlations if c.get('confidence', 0) <= 0.5])
                }
            
            # Analyze alert patterns
            alerts = self.shared_data.get('alerts', [])
            if alerts:
                alert_types = {}
                for alert in alerts:
                    alert_type = alert.get('type', 'unknown')
                    alert_types[alert_type] = alert_types.get(alert_type, 0) + 1
                
                findings['error_analysis']['alert_distribution'] = alert_types
            
            findings['system_behavior']['overall_health'] = (
                'healthy' if len(alerts) == 0 else
                'degraded' if len(alerts) < 5 else
                'critical'
            )
            
        except Exception as e:
            logger.error(f"Unified findings generation error: {e}")
            findings['error'] = str(e)
        
        return findings

    def _generate_unified_recommendations(self) -> List[str]:
        """Generate unified recommendations based on all component analysis"""
        recommendations = []
        
        try:
            # Check if any components failed
            failed_components = [
                name for name, info in self.components.items()
                if not info['running']
            ]
            
            if failed_components:
                recommendations.append(f"Components failed to start: {', '.join(failed_components)} - check system permissions and dependencies")
            
            # Check alert patterns
            alerts = self.shared_data.get('alerts', [])
            if len(alerts) > 10:
                recommendations.append("High number of alerts detected - investigate system stability")
            
            # Check correlation patterns
            correlations = self.shared_data.get('correlations', [])
            high_confidence_correlations = [c for c in correlations if c.get('confidence', 0) > 0.8]
            
            if len(high_confidence_correlations) > 5:
                recommendations.append("Strong correlations detected between components - system showing predictable behavior patterns")
            
            if len(correlations) == 0:
                recommendations.append("No correlations detected - verify components are actively monitoring system events")
            
            # General recommendations
            recommendations.extend([
                "Review component-specific reports for detailed analysis",
                "Use token test sequences to validate DSMIL device responses",
                "Monitor thermal conditions during extended debugging sessions",
                "Correlate findings with Dell MIL-SPEC documentation"
            ])
            
        except Exception as e:
            logger.error(f"Unified recommendations generation error: {e}")
            recommendations.append(f"Error generating recommendations: {e}")
        
        return recommendations

    def _signal_handler(self, signum, frame):
        """Handle system signals gracefully"""
        logger.info(f"Received signal {signum}, stopping debugging session")
        self.stop_unified_debugging()


def main():
    """Main CLI interface for unified debug orchestrator"""
    parser = argparse.ArgumentParser(
        description="Unified DSMIL Debug Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 unified_debug_orchestrator.py --duration 300
  python3 unified_debug_orchestrator.py --test-tokens 0x0480:0x048F
  python3 unified_debug_orchestrator.py --interactive
        """
    )
    
    parser.add_argument("--duration", "-d", type=int, default=300,
                       help="Debugging session duration in seconds (default: 300)")
    parser.add_argument("--output-dir", "-o", type=str, default="/tmp/dsmil_unified_debug",
                       help="Base output directory (default: /tmp/dsmil_unified_debug)")
    parser.add_argument("--test-tokens", "-t", type=str,
                       help="Token range to test (format: 0x0480:0x048F)")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Start interactive debugging session")
    parser.add_argument("--auto-report-interval", type=int, default=60,
                       help="Automatic report generation interval in seconds")
    parser.add_argument("--disable-alerts", action="store_true",
                       help="Disable real-time alerting")
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = UnifiedDebugOrchestrator(args.output_dir)
    
    # Configure session
    orchestrator.configure_session(
        duration_seconds=args.duration,
        auto_report_interval=args.auto_report_interval,
        enable_realtime_alerts=not args.disable_alerts
    )
    
    try:
        if args.interactive:
            # Interactive mode
            print("DSMIL Unified Debug Orchestrator - Interactive Mode")
            print("=" * 60)
            
            orchestrator.start_unified_debugging()
            
            while orchestrator.running:
                try:
                    cmd = input("\nCommands: [t]oken test, [r]eport, [s]tatus, [q]uit: ").strip().lower()
                    
                    if cmd == 'q' or cmd == 'quit':
                        break
                    elif cmd == 't' or cmd == 'token':
                        token_range = input("Token range (e.g., 0x0480:0x048F): ").strip()
                        if ':' in token_range:
                            start_hex, end_hex = token_range.split(':')
                            start_token = int(start_hex, 16)
                            end_token = int(end_hex, 16)
                            results = orchestrator.execute_token_test_sequence(
                                range(start_token, end_token + 1)
                            )
                            print(f"Token test completed: {len(results)} operations")
                        else:
                            print("Invalid token range format")
                    elif cmd == 'r' or cmd == 'report':
                        report_file = orchestrator.generate_unified_report()
                        print(f"Report generated: {report_file}")
                    elif cmd == 's' or cmd == 'status':
                        for name, info in orchestrator.components.items():
                            status = "RUNNING" if info['running'] else "STOPPED"
                            print(f"{name}: {status}")
                    else:
                        print("Unknown command")
                        
                except KeyboardInterrupt:
                    break
                except EOFError:
                    break
                    
        else:
            # Automated mode
            print(f"Starting unified debugging session for {args.duration} seconds")
            
            orchestrator.start_unified_debugging()
            
            # Execute token test if requested
            if args.test_tokens:
                try:
                    if ':' in args.test_tokens:
                        start_hex, end_hex = args.test_tokens.split(':')
                        start_token = int(start_hex, 16)
                        end_token = int(end_hex, 16)
                        
                        print(f"Executing token test sequence: 0x{start_token:04X} to 0x{end_token:04X}")
                        results = orchestrator.execute_token_test_sequence(
                            range(start_token, end_token + 1)
                        )
                        print(f"Token test completed: {len(results)} operations")
                    else:
                        print(f"Invalid token range format: {args.test_tokens}")
                except ValueError as e:
                    print(f"Token range parsing error: {e}")
            
            # Wait for session to complete
            try:
                while orchestrator.running:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nDebugging interrupted by user")
            
            final_report = orchestrator.generate_unified_report()
            print(f"Debugging session complete. Final report: {final_report}")
    
    finally:
        orchestrator.stop_unified_debugging()


if __name__ == "__main__":
    main()