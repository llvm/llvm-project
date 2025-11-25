#!/usr/bin/env python3
"""
Kernel Trace Analyzer for DSMIL Systems

Advanced kernel message tracing and analysis specifically for DSMIL responses.
Provides deep analysis of kernel behavior during token operations.

Dell Latitude 5450 MIL-SPEC - Debian Trixie Compatibility
"""

import re
import sys
import json
import time
import subprocess
import threading
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, List, Set, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KernelTraceAnalyzer:
    """Advanced kernel trace analysis for DSMIL operations"""
    
    def __init__(self, trace_dir: str = "/tmp/dsmil_traces"):
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(exist_ok=True)
        
        # Pattern definitions for kernel message analysis
        self.patterns = {
            # DSMIL-specific patterns
            'dsmil_module': re.compile(r'dsmil[-_]?72dev|DSMIL', re.IGNORECASE),
            'token_access': re.compile(r'token.*0x([0-9a-fA-F]{3,4})', re.IGNORECASE),
            'smbios_call': re.compile(r'dell[-_]?smbios|SMBIOS', re.IGNORECASE),
            'group_activation': re.compile(r'Group\s+(\d+).*activ|activat.*group\s+(\d+)', re.IGNORECASE),
            'device_state': re.compile(r'Device.*(\d+)\.(\d+).*state.*(\w+)|(\w+).*device.*(\d+)\.(\d+)', re.IGNORECASE),
            
            # Memory and mapping patterns  
            'memory_map': re.compile(r'ioremap|iounmap|mapping.*0x([0-9a-fA-F]+)', re.IGNORECASE),
            'chunk_mapping': re.compile(r'chunk.*(\d+).*0x([0-9a-fA-F]+)', re.IGNORECASE),
            'memory_access': re.compile(r'read.*0x([0-9a-fA-F]+)|write.*0x([0-9a-fA-F]+)', re.IGNORECASE),
            
            # ACPI patterns
            'acpi_method': re.compile(r'ACPI.*method.*(\w+)|evaluating.*ACPI.*(\w+)', re.IGNORECASE),
            'acpi_error': re.compile(r'ACPI.*error|error.*ACPI', re.IGNORECASE),
            'acpi_device': re.compile(r'ACPI.*device.*(\w+)|device.*ACPI.*(\w+)', re.IGNORECASE),
            
            # Thermal and power patterns
            'thermal_event': re.compile(r'thermal.*(\d+).*°C|temperature.*(\d+)', re.IGNORECASE),
            'power_event': re.compile(r'power.*state|suspend|resume', re.IGNORECASE),
            
            # Error patterns
            'error_critical': re.compile(r'error|fail|critical|panic|oops|bug', re.IGNORECASE),
            'timeout_event': re.compile(r'timeout|timed.*out', re.IGNORECASE),
            
            # Hardware patterns
            'pci_event': re.compile(r'PCI.*device|pci.*bus', re.IGNORECASE),
            'hardware_error': re.compile(r'hardware.*error|MCE|machine.*check', re.IGNORECASE)
        }
        
        # Trace data storage
        self.traces = deque(maxlen=50000)  # Store up to 50k trace entries
        self.pattern_matches = defaultdict(list)
        self.correlation_data = defaultdict(list)
        
        # Analysis state
        self.baseline_established = False
        self.baseline_patterns = {}
        self.anomaly_threshold = 3.0  # Standard deviations for anomaly detection
        
        # Statistics
        self.stats = {
            'total_messages': 0,
            'dsmil_messages': 0,
            'token_accesses': 0,
            'errors': 0,
            'anomalies': 0
        }

    def start_realtime_tracing(self, duration_seconds: Optional[int] = None):
        """Start real-time kernel tracing"""
        logger.info("Starting real-time kernel tracing")
        
        # Use journalctl for kernel message streaming
        cmd = ['journalctl', '-f', '-k', '--no-pager', '--output=json']
        
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                text=True, bufsize=1
            )
            
            start_time = datetime.now()
            
            while proc.poll() is None:
                # Check duration limit
                if duration_seconds and (datetime.now() - start_time).total_seconds() > duration_seconds:
                    break
                
                line = proc.stdout.readline()
                if not line:
                    continue
                
                try:
                    # Parse JSON output from journalctl
                    entry = json.loads(line.strip())
                    self._process_journal_entry(entry)
                except json.JSONDecodeError:
                    # Fallback to plain text processing
                    self._process_plain_message(line.strip())
                except KeyboardInterrupt:
                    logger.info("Tracing interrupted by user")
                    break
                    
        except FileNotFoundError:
            logger.error("journalctl not found, falling back to dmesg")
            self._fallback_dmesg_tracing(duration_seconds)
        finally:
            if 'proc' in locals() and proc.poll() is None:
                proc.terminate()
                proc.wait()

    def _process_journal_entry(self, entry: Dict):
        """Process structured journal entry"""
        try:
            timestamp_us = int(entry.get('__REALTIME_TIMESTAMP', 0))
            timestamp = datetime.fromtimestamp(timestamp_us / 1000000)
            
            message = entry.get('MESSAGE', '')
            priority = entry.get('PRIORITY', '6')  # Default to info
            syslog_id = entry.get('SYSLOG_IDENTIFIER', 'kernel')
            
            # Only process kernel messages
            if syslog_id != 'kernel' and '_TRANSPORT' in entry and entry['_TRANSPORT'] != 'kernel':
                return
            
            trace_entry = {
                'timestamp': timestamp,
                'message': message,
                'priority': priority,
                'raw_entry': entry
            }
            
            self._analyze_message(trace_entry)
            
        except Exception as e:
            logger.debug(f"Failed to process journal entry: {e}")

    def _process_plain_message(self, line: str):
        """Process plain text message"""
        try:
            # Simple timestamp extraction for non-JSON format
            timestamp = datetime.now()
            
            trace_entry = {
                'timestamp': timestamp,
                'message': line,
                'priority': '6',
                'raw_entry': {'MESSAGE': line}
            }
            
            self._analyze_message(trace_entry)
            
        except Exception as e:
            logger.debug(f"Failed to process plain message: {e}")

    def _analyze_message(self, trace_entry: Dict):
        """Analyze individual trace message"""
        message = trace_entry['message']
        timestamp = trace_entry['timestamp']
        
        # Store trace entry
        self.traces.append(trace_entry)
        self.stats['total_messages'] += 1
        
        # Pattern matching analysis
        matches = {}
        for pattern_name, pattern in self.patterns.items():
            match = pattern.search(message)
            if match:
                matches[pattern_name] = {
                    'match': match.groups() if match.groups() else [match.group(0)],
                    'full_match': match.group(0)
                }
                
                # Store pattern match with context
                self.pattern_matches[pattern_name].append({
                    'timestamp': timestamp,
                    'message': message,
                    'match_data': matches[pattern_name]
                })
        
        # Specific analysis for DSMIL messages
        if any(name.startswith('dsmil_') or name.startswith('token_') for name in matches):
            self.stats['dsmil_messages'] += 1
            self._analyze_dsmil_specific(trace_entry, matches)
        
        # Error detection
        if 'error_critical' in matches or 'timeout_event' in matches:
            self.stats['errors'] += 1
            self._analyze_error(trace_entry, matches)
        
        # Real-time anomaly detection
        if self.baseline_established:
            if self._detect_anomaly(trace_entry, matches):
                self.stats['anomalies'] += 1
        
        # Output interesting findings in real-time
        if matches and any(name in ['dsmil_module', 'token_access', 'group_activation'] for name in matches):
            logger.info(f"DSMIL Activity: {message[:100]}...")

    def _analyze_dsmil_specific(self, trace_entry: Dict, matches: Dict):
        """Analyze DSMIL-specific messages"""
        message = trace_entry['message']
        timestamp = trace_entry['timestamp']
        
        # Token access analysis
        if 'token_access' in matches:
            token_hex = matches['token_access']['match'][0]
            try:
                token_id = int(token_hex, 16)
                if 0x0480 <= token_id <= 0x04C7:  # Our target range
                    self.stats['token_accesses'] += 1
                    group_id = (token_id - 0x0480) // 12
                    device_id = (token_id - 0x0480) % 12
                    
                    logger.info(f"Token 0x{token_id:04X} accessed (Group {group_id}, Device {device_id})")
                    
                    # Store detailed token access info
                    self.correlation_data['token_accesses'].append({
                        'timestamp': timestamp,
                        'token_id': token_id,
                        'group_id': group_id,
                        'device_id': device_id,
                        'message': message
                    })
            except ValueError:
                pass
        
        # Group activation analysis
        if 'group_activation' in matches:
            group_match = matches['group_activation']['match']
            group_id = next((g for g in group_match if g and g.isdigit()), None)
            if group_id:
                logger.info(f"Group {group_id} activation detected")
                self.correlation_data['group_activations'].append({
                    'timestamp': timestamp,
                    'group_id': int(group_id),
                    'message': message
                })

    def _analyze_error(self, trace_entry: Dict, matches: Dict):
        """Analyze error messages"""
        message = trace_entry['message']
        timestamp = trace_entry['timestamp']
        
        error_info = {
            'timestamp': timestamp,
            'message': message,
            'error_types': list(matches.keys()),
            'severity': 'high' if 'critical' in message.lower() or 'panic' in message.lower() else 'medium'
        }
        
        self.correlation_data['errors'].append(error_info)
        logger.warning(f"Error detected: {message[:100]}...")

    def _detect_anomaly(self, trace_entry: Dict, matches: Dict) -> bool:
        """Detect anomalies based on baseline patterns"""
        # Simple anomaly detection based on pattern frequency
        # In a real implementation, this would be more sophisticated
        
        current_hour = trace_entry['timestamp'].hour
        
        for pattern_name in matches:
            # Get baseline frequency for this pattern at this hour
            baseline_freq = self.baseline_patterns.get(f"{pattern_name}_{current_hour}", 0)
            
            # Count recent occurrences
            recent_count = len([
                m for m in self.pattern_matches[pattern_name][-100:]  # Last 100 matches
                if m['timestamp'].hour == current_hour
            ])
            
            # Simple threshold-based anomaly detection
            if baseline_freq > 0 and recent_count > baseline_freq * self.anomaly_threshold:
                logger.warning(f"Anomaly detected: {pattern_name} frequency {recent_count} vs baseline {baseline_freq}")
                return True
        
        return False

    def _fallback_dmesg_tracing(self, duration_seconds: Optional[int] = None):
        """Fallback to dmesg-based tracing"""
        logger.info("Using dmesg for kernel message tracing")
        
        start_time = datetime.now()
        last_messages = set()
        
        try:
            while True:
                if duration_seconds and (datetime.now() - start_time).total_seconds() > duration_seconds:
                    break
                
                # Get recent dmesg output
                result = subprocess.run(['dmesg', '-T'], capture_output=True, text=True)
                if result.returncode != 0:
                    logger.error("Failed to run dmesg")
                    break
                
                lines = result.stdout.strip().split('\n')
                
                # Process new messages only
                for line in lines:
                    if line not in last_messages:
                        self._process_plain_message(line)
                
                last_messages = set(lines[-1000:])  # Keep track of last 1000 messages
                time.sleep(1)  # Check every second
                
        except KeyboardInterrupt:
            logger.info("Dmesg tracing interrupted")

    def establish_baseline(self, duration_minutes: int = 10):
        """Establish baseline patterns for anomaly detection"""
        logger.info(f"Establishing baseline over {duration_minutes} minutes")
        
        baseline_start = datetime.now()
        baseline_patterns = defaultdict(lambda: defaultdict(int))
        
        # Clear existing data
        self.traces.clear()
        self.pattern_matches.clear()
        
        # Collect baseline data
        self.start_realtime_tracing(duration_minutes * 60)
        
        # Analyze collected data to create baseline
        for pattern_name, matches in self.pattern_matches.items():
            for match in matches:
                hour = match['timestamp'].hour
                baseline_patterns[f"{pattern_name}_{hour}"] += 1
        
        # Store baseline patterns
        self.baseline_patterns = dict(baseline_patterns)
        self.baseline_established = True
        
        logger.info(f"Baseline established with {len(self.baseline_patterns)} pattern-hour combinations")
        
        # Save baseline to file
        baseline_file = self.trace_dir / f"baseline_{int(baseline_start.timestamp())}.json"
        with open(baseline_file, 'w') as f:
            json.dump({
                'timestamp': baseline_start.isoformat(),
                'duration_minutes': duration_minutes,
                'patterns': self.baseline_patterns,
                'total_messages': self.stats['total_messages']
            }, f, indent=2)

    def analyze_token_sequences(self) -> Dict[str, List]:
        """Analyze token access sequences and patterns"""
        token_accesses = self.correlation_data.get('token_accesses', [])
        
        if not token_accesses:
            return {"error": "No token accesses recorded"}
        
        # Sort by timestamp
        sorted_accesses = sorted(token_accesses, key=lambda x: x['timestamp'])
        
        # Analyze sequences
        sequences = []
        current_sequence = []
        sequence_timeout = timedelta(seconds=5)  # Max gap between accesses in same sequence
        
        for access in sorted_accesses:
            if current_sequence and (access['timestamp'] - current_sequence[-1]['timestamp']) > sequence_timeout:
                # End current sequence, start new one
                if len(current_sequence) > 1:
                    sequences.append(current_sequence)
                current_sequence = [access]
            else:
                current_sequence.append(access)
        
        # Add final sequence if it exists
        if len(current_sequence) > 1:
            sequences.append(current_sequence)
        
        # Analyze group patterns
        group_patterns = defaultdict(int)
        for access in sorted_accesses:
            group_patterns[access['group_id']] += 1
        
        # Analyze temporal patterns
        hourly_distribution = defaultdict(int)
        for access in sorted_accesses:
            hourly_distribution[access['timestamp'].hour] += 1
        
        return {
            "total_accesses": len(sorted_accesses),
            "sequences": [
                {
                    "length": len(seq),
                    "duration": (seq[-1]['timestamp'] - seq[0]['timestamp']).total_seconds(),
                    "tokens": [f"0x{a['token_id']:04X}" for a in seq],
                    "groups": list(set(a['group_id'] for a in seq))
                }
                for seq in sequences
            ],
            "group_distribution": dict(group_patterns),
            "hourly_distribution": dict(hourly_distribution),
            "unique_tokens": len(set(a['token_id'] for a in sorted_accesses))
        }

    def generate_trace_report(self) -> str:
        """Generate comprehensive trace analysis report"""
        report_file = self.trace_dir / f"trace_report_{int(datetime.now().timestamp())}.json"
        
        # Analyze recent activity (last hour)
        one_hour_ago = datetime.now() - timedelta(hours=1)
        recent_traces = [t for t in self.traces if t['timestamp'] > one_hour_ago]
        
        # Pattern frequency analysis
        pattern_frequency = {}
        for pattern_name, matches in self.pattern_matches.items():
            recent_matches = [m for m in matches if m['timestamp'] > one_hour_ago]
            pattern_frequency[pattern_name] = {
                'total': len(matches),
                'recent': len(recent_matches)
            }
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "analysis_period": "last_hour",
            "statistics": self.stats,
            "pattern_frequency": pattern_frequency,
            "token_analysis": self.analyze_token_sequences(),
            "recent_errors": [
                {
                    "timestamp": err['timestamp'].isoformat(),
                    "message": err['message'][:200],
                    "severity": err['severity']
                }
                for err in self.correlation_data.get('errors', [])[-10:]
            ],
            "system_health": {
                "baseline_established": self.baseline_established,
                "anomalies_detected": self.stats.get('anomalies', 0),
                "trace_buffer_usage": f"{len(self.traces)}/50000"
            },
            "recommendations": self._generate_recommendations()
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Trace report generated: {report_file}")
        return str(report_file)

    def _generate_recommendations(self) -> List[str]:
        """Generate analysis-based recommendations"""
        recommendations = []
        
        if self.stats['errors'] > 0:
            recommendations.append("Errors detected - review error log for system stability")
        
        if self.stats['dsmil_messages'] == 0:
            recommendations.append("No DSMIL activity detected - verify module is loaded and active")
        
        if self.stats['token_accesses'] > 100:
            recommendations.append("High token access frequency - monitor for potential issues")
        
        if not self.baseline_established:
            recommendations.append("Establish baseline patterns for better anomaly detection")
        
        return recommendations


def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="DSMIL Kernel Trace Analyzer")
    parser.add_argument("--baseline", "-b", type=int, default=0,
                       help="Establish baseline for N minutes")
    parser.add_argument("--trace", "-t", type=int, default=300,
                       help="Trace for N seconds")
    parser.add_argument("--output-dir", "-o", type=str, default="/tmp/dsmil_traces",
                       help="Output directory")
    parser.add_argument("--token-focus", "-f", action="store_true",
                       help="Focus on token-related messages only")
    
    args = parser.parse_args()
    
    analyzer = KernelTraceAnalyzer(args.output_dir)
    
    try:
        if args.baseline > 0:
            analyzer.establish_baseline(args.baseline)
        
        print(f"Starting trace analysis for {args.trace} seconds...")
        analyzer.start_realtime_tracing(args.trace)
        
        # Generate final report
        report_file = analyzer.generate_trace_report()
        print(f"Analysis complete. Report: {report_file}")
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        report_file = analyzer.generate_trace_report()
        print(f"Partial report: {report_file}")


if __name__ == "__main__":
    main()