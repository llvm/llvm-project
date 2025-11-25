#!/usr/bin/env python3
"""
DSMIL Response Correlation System
=================================

Advanced correlation system for analyzing DSMIL kernel module responses
to SMBIOS token activation on Dell Latitude 5450 MIL-SPEC systems.

Features:
- Real-time DSMIL kernel module response monitoring
- Token activation to DSMIL device response correlation
- Memory mapping analysis and device state tracking
- Pattern recognition for DSMIL device group activation
- Response timing and thermal correlation analysis
- Integration with existing monitoring infrastructure

Target System: Dell Latitude 5450 MIL-SPEC (72 DSMIL devices)
Token Range: 0x0480-0x04C7 (72 tokens mapped to 6 groups × 12 devices)

Author: TESTBED Agent
Version: 1.0.0
Date: 2025-09-01
"""

import os
import sys
import time
import json
import re
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum

try:
    import psutil
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "--user", "psutil"], check=True)
    import psutil

class DSMILResponseType(Enum):
    """Types of DSMIL responses"""
    DEVICE_ACTIVATION = "DEVICE_ACTIVATION"
    DEVICE_DEACTIVATION = "DEVICE_DEACTIVATION"
    MEMORY_MAPPING = "MEMORY_MAPPING"
    GROUP_RESPONSE = "GROUP_RESPONSE"
    ERROR_CONDITION = "ERROR_CONDITION"
    THERMAL_EVENT = "THERMAL_EVENT"
    SIGNATURE_DETECTION = "SIGNATURE_DETECTION"

@dataclass
class DSMILResponse:
    """Individual DSMIL response record"""
    timestamp: datetime
    response_type: DSMILResponseType
    device_id: Optional[int] = None
    group_id: Optional[int] = None
    memory_address: Optional[str] = None
    message: str = ""
    raw_data: str = ""
    related_token: Optional[str] = None
    thermal_reading: Optional[float] = None
    correlation_confidence: float = 0.0

@dataclass
class TokenResponseCorrelation:
    """Correlation between token activation and DSMIL responses"""
    token: str
    activation_time: datetime
    responses: List[DSMILResponse] = field(default_factory=list)
    response_delay: Optional[float] = None  # Seconds from activation to first response
    affected_devices: Set[int] = field(default_factory=set)
    affected_groups: Set[int] = field(default_factory=set)
    correlation_strength: float = 0.0
    pattern_signature: Optional[str] = None

class DSMILResponseCorrelator:
    """Main DSMIL response correlation system"""
    
    def __init__(self, work_dir: str = "/home/john/LAT5150DRVMIL"):
        self.work_dir = Path(work_dir)
        self.correlation_dir = self.work_dir / "testing" / "correlations"
        self.correlation_dir.mkdir(parents=True, exist_ok=True)
        
        # Response monitoring
        self.response_buffer: deque = deque(maxlen=1000)  # Keep last 1000 responses
        self.correlation_window = 30.0  # 30 second correlation window
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Token tracking
        self.active_tokens: Dict[str, datetime] = {}
        self.completed_correlations: List[TokenResponseCorrelation] = []
        
        # DSMIL device mapping (based on kernel module structure)
        self.device_groups = {
            0: list(range(0, 12)),   # Group 0: devices 0-11
            1: list(range(12, 24)),  # Group 1: devices 12-23
            2: list(range(24, 36)),  # Group 2: devices 24-35
            3: list(range(36, 48)),  # Group 3: devices 36-47
            4: list(range(48, 60)),  # Group 4: devices 48-59
            5: list(range(60, 72))   # Group 5: devices 60-71
        }
        
        # Token to group mapping (based on target ranges)
        self.token_to_group = {}
        for group_id in range(6):
            for i, token_offset in enumerate(range(12)):  # 12 tokens per group
                token_hex = 0x0480 + (group_id * 12) + token_offset
                self.token_to_group[f"0x{token_hex:04X}"] = group_id
                
        # Response patterns
        self.response_patterns = {
            'device_activation': [
                r'DSMIL device (\d+) activated',
                r'Device (\d+): status changed to active',
                r'dsmil_72dev: Device (\d+) responding'
            ],
            'memory_mapping': [
                r'Memory mapped at (0x[0-9a-fA-F]+)',
                r'DSMIL memory chunk (\d+) mapped',
                r'Virtual address (0x[0-9a-fA-F]+) mapped'
            ],
            'group_response': [
                r'Group (\d+) devices responding',
                r'DSMIL group (\d+): (\d+) devices active',
                r'Group (\d+) signature detected'
            ],
            'error_condition': [
                r'DSMIL error: (.+)',
                r'Device (\d+) error: (.+)',
                r'Memory mapping failed: (.+)'
            ]
        }
        
    def start_response_monitoring(self):
        """Start continuous DSMIL response monitoring"""
        
        print("🔍 Starting DSMIL response monitoring...")
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        print("✅ DSMIL response monitoring started")
        
    def stop_response_monitoring(self):
        """Stop DSMIL response monitoring"""
        
        print("🛑 Stopping DSMIL response monitoring...")
        
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
            
        print("✅ DSMIL response monitoring stopped")
        
    def _monitor_loop(self):
        """Main monitoring loop"""
        
        last_dmesg_lines = 0
        
        while self.monitoring_active:
            try:
                # Monitor dmesg for kernel messages
                responses = self._check_dmesg_responses(last_dmesg_lines)
                
                for response in responses:
                    self.response_buffer.append(response)
                    
                # Monitor DSMIL sysfs entries
                sysfs_responses = self._check_sysfs_responses()
                for response in sysfs_responses:
                    self.response_buffer.append(response)
                    
                # Monitor memory mapping changes
                memory_responses = self._check_memory_mapping_changes()
                for response in memory_responses:
                    self.response_buffer.append(response)
                    
                # Update last dmesg position
                last_dmesg_lines = self._get_dmesg_line_count()
                
                # Process correlations for recently activated tokens
                self._process_pending_correlations()
                
                time.sleep(1)  # Check every second
                
            except Exception as e:
                print(f"⚠️ Monitoring error: {e}")
                time.sleep(5)  # Retry delay
                
    def _check_dmesg_responses(self, last_lines: int) -> List[DSMILResponse]:
        """Check dmesg for new DSMIL-related messages"""
        
        responses = []
        
        try:
            # Get recent dmesg entries
            result = subprocess.run([
                'dmesg', '--time-format=iso', '--level=info,notice,warn,err'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                return responses
                
            lines = result.stdout.strip().split('\n')
            new_lines = lines[last_lines:] if last_lines > 0 else lines[-50:]
            
            for line in new_lines:
                if not line.strip():
                    continue
                    
                # Check for DSMIL-related messages
                if 'dsmil' in line.lower():
                    response = self._parse_dmesg_line(line)
                    if response:
                        responses.append(response)
                        
        except Exception as e:
            print(f"⚠️ dmesg check error: {e}")
            
        return responses
        
    def _parse_dmesg_line(self, line: str) -> Optional[DSMILResponse]:
        """Parse a dmesg line for DSMIL information"""
        
        try:
            # Extract timestamp (ISO format from dmesg)
            timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2},\d+[+-]\d{4})', line)
            if timestamp_match:
                timestamp_str = timestamp_match.group(1)
                # Convert to datetime (may need adjustment for timezone)
                timestamp = datetime.now(timezone.utc)  # Simplified for now
            else:
                timestamp = datetime.now(timezone.utc)
                
            # Check each pattern type
            for pattern_type, patterns in self.response_patterns.items():
                for pattern in patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        return self._create_response_from_pattern(
                            timestamp, pattern_type, match, line
                        )
                        
            # General DSMIL message
            if 'dsmil' in line.lower():
                return DSMILResponse(
                    timestamp=timestamp,
                    response_type=DSMILResponseType.SIGNATURE_DETECTION,
                    message=line.strip(),
                    raw_data=line
                )
                
        except Exception as e:
            print(f"⚠️ dmesg parsing error: {e}")
            
        return None
        
    def _create_response_from_pattern(self, timestamp: datetime, pattern_type: str, 
                                    match: re.Match, raw_line: str) -> DSMILResponse:
        """Create DSMIL response from pattern match"""
        
        if pattern_type == 'device_activation':
            device_id = int(match.group(1)) if match.group(1) else None
            group_id = self._device_to_group(device_id) if device_id else None
            
            return DSMILResponse(
                timestamp=timestamp,
                response_type=DSMILResponseType.DEVICE_ACTIVATION,
                device_id=device_id,
                group_id=group_id,
                message=f"Device {device_id} activated",
                raw_data=raw_line,
                correlation_confidence=0.8
            )
            
        elif pattern_type == 'memory_mapping':
            memory_addr = match.group(1) if match.group(1) else None
            
            return DSMILResponse(
                timestamp=timestamp,
                response_type=DSMILResponseType.MEMORY_MAPPING,
                memory_address=memory_addr,
                message=f"Memory mapped at {memory_addr}",
                raw_data=raw_line,
                correlation_confidence=0.6
            )
            
        elif pattern_type == 'group_response':
            group_id = int(match.group(1)) if match.group(1) else None
            
            return DSMILResponse(
                timestamp=timestamp,
                response_type=DSMILResponseType.GROUP_RESPONSE,
                group_id=group_id,
                message=f"Group {group_id} response",
                raw_data=raw_line,
                correlation_confidence=0.9
            )
            
        elif pattern_type == 'error_condition':
            error_msg = match.group(1) if len(match.groups()) > 0 else "Unknown error"
            
            return DSMILResponse(
                timestamp=timestamp,
                response_type=DSMILResponseType.ERROR_CONDITION,
                message=f"Error: {error_msg}",
                raw_data=raw_line,
                correlation_confidence=0.7
            )
            
        else:
            return DSMILResponse(
                timestamp=timestamp,
                response_type=DSMILResponseType.SIGNATURE_DETECTION,
                message="Pattern detected",
                raw_data=raw_line,
                correlation_confidence=0.5
            )
            
    def _check_sysfs_responses(self) -> List[DSMILResponse]:
        """Check DSMIL sysfs entries for device state changes"""
        
        responses = []
        
        try:
            # Check for DSMIL module sysfs entries
            dsmil_sysfs = Path("/sys/module/dsmil_72dev")
            if dsmil_sysfs.exists():
                # Check device status files
                for device_dir in dsmil_sysfs.glob("devices/*/"):
                    device_id = self._extract_device_id(device_dir.name)
                    
                    status_file = device_dir / "status"
                    if status_file.exists():
                        try:
                            status = status_file.read_text().strip()
                            if status not in ["inactive", "unknown"]:
                                response = DSMILResponse(
                                    timestamp=datetime.now(timezone.utc),
                                    response_type=DSMILResponseType.DEVICE_ACTIVATION,
                                    device_id=device_id,
                                    group_id=self._device_to_group(device_id),
                                    message=f"Device {device_id} status: {status}",
                                    raw_data=f"sysfs:{device_dir}/status={status}",
                                    correlation_confidence=0.7
                                )
                                responses.append(response)
                        except OSError:
                            pass
                            
        except Exception as e:
            print(f"⚠️ sysfs check error: {e}")
            
        return responses
        
    def _check_memory_mapping_changes(self) -> List[DSMILResponse]:
        """Check for memory mapping changes in /proc/iomem"""
        
        responses = []
        
        try:
            # Check /proc/iomem for DSMIL memory regions
            with open('/proc/iomem', 'r') as f:
                for line in f:
                    if 'dsmil' in line.lower() or '52000000' in line:
                        # Parse memory range
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            memory_range = parts[0]
                            description = ' '.join(parts[1:])
                            
                            response = DSMILResponse(
                                timestamp=datetime.now(timezone.utc),
                                response_type=DSMILResponseType.MEMORY_MAPPING,
                                memory_address=memory_range,
                                message=f"Memory mapping: {description}",
                                raw_data=line.strip(),
                                correlation_confidence=0.6
                            )
                            responses.append(response)
                            
        except Exception as e:
            print(f"⚠️ Memory mapping check error: {e}")
            
        return responses
        
    def _device_to_group(self, device_id: int) -> Optional[int]:
        """Map device ID to group ID"""
        
        if device_id is None:
            return None
            
        for group_id, devices in self.device_groups.items():
            if device_id in devices:
                return group_id
                
        return None
        
    def _extract_device_id(self, device_name: str) -> Optional[int]:
        """Extract device ID from device name"""
        
        match = re.search(r'device_?(\d+)', device_name)
        return int(match.group(1)) if match else None
        
    def _get_dmesg_line_count(self) -> int:
        """Get current number of dmesg lines"""
        
        try:
            result = subprocess.run(['dmesg'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return len(result.stdout.strip().split('\n'))
        except:
            pass
            
        return 0
        
    def register_token_activation(self, token: str):
        """Register token activation for correlation tracking"""
        
        self.active_tokens[token] = datetime.now(timezone.utc)
        print(f"🔗 Registered token activation: {token}")
        
    def _process_pending_correlations(self):
        """Process correlations for recently activated tokens"""
        
        current_time = datetime.now(timezone.utc)
        completed_tokens = []
        
        for token, activation_time in self.active_tokens.items():
            time_elapsed = (current_time - activation_time).total_seconds()
            
            if time_elapsed >= self.correlation_window:
                # Process correlation for this token
                correlation = self._create_token_correlation(token, activation_time)
                self.completed_correlations.append(correlation)
                completed_tokens.append(token)
                
                print(f"📊 Completed correlation for token {token}: "
                      f"{len(correlation.responses)} responses, "
                      f"strength {correlation.correlation_strength:.2f}")
                      
        # Remove completed tokens
        for token in completed_tokens:
            del self.active_tokens[token]
            
    def _create_token_correlation(self, token: str, activation_time: datetime) -> TokenResponseCorrelation:
        """Create correlation between token and responses"""
        
        correlation = TokenResponseCorrelation(
            token=token,
            activation_time=activation_time
        )
        
        # Find responses within correlation window
        window_start = activation_time
        window_end = activation_time + datetime.timedelta(seconds=self.correlation_window)
        
        relevant_responses = []
        for response in self.response_buffer:
            if window_start <= response.timestamp <= window_end:
                # Additional filtering for relevance
                if self._is_response_relevant_to_token(token, response):
                    relevant_responses.append(response)
                    
        # Sort by timestamp
        relevant_responses.sort(key=lambda r: r.timestamp)
        correlation.responses = relevant_responses
        
        if relevant_responses:
            # Calculate response delay
            first_response_time = relevant_responses[0].timestamp
            correlation.response_delay = (first_response_time - activation_time).total_seconds()
            
            # Collect affected devices and groups
            for response in relevant_responses:
                if response.device_id is not None:
                    correlation.affected_devices.add(response.device_id)
                if response.group_id is not None:
                    correlation.affected_groups.add(response.group_id)
                    
            # Calculate correlation strength
            correlation.correlation_strength = self._calculate_correlation_strength(token, relevant_responses)
            
            # Generate pattern signature
            correlation.pattern_signature = self._generate_pattern_signature(relevant_responses)
            
        return correlation
        
    def _is_response_relevant_to_token(self, token: str, response: DSMILResponse) -> bool:
        """Determine if response is relevant to token"""
        
        # Check if token maps to same group as response
        expected_group = self.token_to_group.get(token)
        if expected_group is not None and response.group_id == expected_group:
            return True
            
        # Check if response device is in expected group
        if expected_group is not None and response.device_id is not None:
            if response.device_id in self.device_groups.get(expected_group, []):
                return True
                
        # Check for memory mapping in DSMIL range
        if response.response_type == DSMILResponseType.MEMORY_MAPPING:
            if response.memory_address and '52000000' in response.memory_address:
                return True
                
        # General DSMIL activity during token window
        if response.response_type in [
            DSMILResponseType.SIGNATURE_DETECTION,
            DSMILResponseType.DEVICE_ACTIVATION
        ]:
            return True
            
        return False
        
    def _calculate_correlation_strength(self, token: str, responses: List[DSMILResponse]) -> float:
        """Calculate correlation strength between token and responses"""
        
        if not responses:
            return 0.0
            
        strength_factors = []
        
        # Factor 1: Response timing (closer to activation = higher strength)
        timing_scores = []
        for response in responses:
            if response.timestamp:
                delay = abs((response.timestamp - self.active_tokens.get(token, datetime.now(timezone.utc))).total_seconds())
                timing_score = max(0, 1.0 - (delay / self.correlation_window))
                timing_scores.append(timing_score)
                
        if timing_scores:
            strength_factors.append(sum(timing_scores) / len(timing_scores))
            
        # Factor 2: Response relevance (group/device matching)
        expected_group = self.token_to_group.get(token)
        relevance_score = 0.0
        
        if expected_group is not None:
            relevant_responses = sum(1 for r in responses 
                                   if r.group_id == expected_group or 
                                      (r.device_id and r.device_id in self.device_groups.get(expected_group, [])))
            relevance_score = relevant_responses / len(responses)
            
        strength_factors.append(relevance_score)
        
        # Factor 3: Response confidence
        confidence_scores = [r.correlation_confidence for r in responses if r.correlation_confidence > 0]
        if confidence_scores:
            strength_factors.append(sum(confidence_scores) / len(confidence_scores))
            
        # Factor 4: Response diversity (different types = more comprehensive)
        response_types = set(r.response_type for r in responses)
        diversity_score = min(1.0, len(response_types) / 3.0)  # Normalize by expected types
        strength_factors.append(diversity_score)
        
        # Calculate weighted average
        return sum(strength_factors) / len(strength_factors) if strength_factors else 0.0
        
    def _generate_pattern_signature(self, responses: List[DSMILResponse]) -> str:
        """Generate pattern signature from responses"""
        
        if not responses:
            return "NO_RESPONSE"
            
        # Create signature from response types and IDs
        signature_parts = []
        
        for response in responses:
            part = response.response_type.value
            
            if response.group_id is not None:
                part += f"_G{response.group_id}"
                
            if response.device_id is not None:
                part += f"_D{response.device_id}"
                
            signature_parts.append(part)
            
        return "|".join(signature_parts[:5])  # Limit to first 5 responses
        
    def get_correlation_for_token(self, token: str) -> Optional[TokenResponseCorrelation]:
        """Get correlation data for specific token"""
        
        for correlation in self.completed_correlations:
            if correlation.token == token:
                return correlation
                
        return None
        
    def get_group_correlations(self, group_id: int) -> List[TokenResponseCorrelation]:
        """Get all correlations for a specific DSMIL group"""
        
        group_correlations = []
        
        for correlation in self.completed_correlations:
            if group_id in correlation.affected_groups:
                group_correlations.append(correlation)
                
        return group_correlations
        
    def save_correlations(self, filename: Optional[str] = None) -> Path:
        """Save correlation data to file"""
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"dsmil_correlations_{timestamp}.json"
            
        filepath = self.correlation_dir / filename
        
        # Convert correlations to serializable format
        correlations_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'total_correlations': len(self.completed_correlations),
            'correlation_window': self.correlation_window,
            'correlations': []
        }
        
        for correlation in self.completed_correlations:
            correlation_dict = {
                'token': correlation.token,
                'activation_time': correlation.activation_time.isoformat(),
                'response_delay': correlation.response_delay,
                'affected_devices': list(correlation.affected_devices),
                'affected_groups': list(correlation.affected_groups),
                'correlation_strength': correlation.correlation_strength,
                'pattern_signature': correlation.pattern_signature,
                'responses': []
            }
            
            # Add response data
            for response in correlation.responses:
                response_dict = {
                    'timestamp': response.timestamp.isoformat(),
                    'response_type': response.response_type.value,
                    'device_id': response.device_id,
                    'group_id': response.group_id,
                    'memory_address': response.memory_address,
                    'message': response.message,
                    'correlation_confidence': response.correlation_confidence
                }
                correlation_dict['responses'].append(response_dict)
                
            correlations_data['correlations'].append(correlation_dict)
            
        with open(filepath, 'w') as f:
            json.dump(correlations_data, f, indent=2)
            
        return filepath
        
    def generate_correlation_report(self) -> str:
        """Generate human-readable correlation report"""
        
        lines = []
        lines.append("=" * 80)
        lines.append("DSMIL RESPONSE CORRELATION REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        # Summary
        lines.append(f"Total Token Correlations: {len(self.completed_correlations)}")
        lines.append(f"Correlation Window: {self.correlation_window} seconds")
        lines.append(f"Currently Active Tokens: {len(self.active_tokens)}")
        lines.append("")
        
        # Group analysis
        group_stats = defaultdict(list)
        for correlation in self.completed_correlations:
            for group_id in correlation.affected_groups:
                group_stats[group_id].append(correlation)
                
        if group_stats:
            lines.append("GROUP RESPONSE ANALYSIS:")
            lines.append("-" * 40)
            
            for group_id in sorted(group_stats.keys()):
                correlations = group_stats[group_id]
                avg_strength = sum(c.correlation_strength for c in correlations) / len(correlations)
                avg_delay = sum(c.response_delay for c in correlations if c.response_delay) / len([c for c in correlations if c.response_delay])
                
                lines.append(f"Group {group_id}:")
                lines.append(f"  Correlations: {len(correlations)}")
                lines.append(f"  Avg Strength: {avg_strength:.2f}")
                lines.append(f"  Avg Delay: {avg_delay:.2f}s")
                lines.append("")
                
        # Strong correlations
        strong_correlations = [c for c in self.completed_correlations if c.correlation_strength >= 0.7]
        if strong_correlations:
            lines.append("STRONG CORRELATIONS (≥0.7):")
            lines.append("-" * 40)
            
            for correlation in strong_correlations:
                lines.append(f"Token {correlation.token}: {correlation.correlation_strength:.2f}")
                lines.append(f"  Response Delay: {correlation.response_delay:.2f}s")
                lines.append(f"  Pattern: {correlation.pattern_signature}")
                lines.append(f"  Groups: {sorted(correlation.affected_groups)}")
                lines.append(f"  Devices: {sorted(correlation.affected_devices)}")
                lines.append("")
                
        return "\n".join(lines)

def main():
    """Test DSMIL response correlation system"""
    
    print("🔗 DSMIL Response Correlation System v1.0.0")
    print("Dell Latitude 5450 MIL-SPEC - TESTBED Agent")
    print("=" * 60)
    
    correlator = DSMILResponseCorrelator()
    
    print("\nStarting response monitoring...")
    correlator.start_response_monitoring()
    
    try:
        # Simulate token activation for testing
        print("\nSimulating token activations for testing...")
        
        test_tokens = ["0x0480", "0x0481", "0x048C"]  # Test tokens from different groups
        
        for token in test_tokens:
            print(f"Activating token {token}...")
            correlator.register_token_activation(token)
            time.sleep(2)  # Brief delay between activations
            
        print("\nMonitoring responses for 30 seconds...")
        time.sleep(35)  # Wait for correlation window + processing time
        
        # Generate report
        report = correlator.generate_correlation_report()
        print(report)
        
        # Save correlations
        correlation_file = correlator.save_correlations()
        print(f"\n📊 Correlations saved to: {correlation_file}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Monitoring interrupted by user")
        
    finally:
        correlator.stop_response_monitoring()
        
    return 0

if __name__ == "__main__":
    sys.exit(main())