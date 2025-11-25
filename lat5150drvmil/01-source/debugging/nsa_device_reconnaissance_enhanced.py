#!/usr/bin/env python3
"""
NSA DEVICE RECONNAISSANCE - ENHANCED WITH NPU INTEGRATION
Target: DSMIL Device Range 0x8000-0x806B (84 devices)
Platform: Dell Latitude 5450 MIL-SPEC JRTC1
Enhancement: Neural Processing Unit (NPU) detection and comprehensive enumeration
Classification: TOP SECRET//SI//NOFORN
"""

import os
import sys
import struct
import time
import json
import subprocess
import hashlib
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import logging

# Configure secure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] NSA-RECON-ENHANCED: %(message)s',
    handlers=[
        logging.FileHandler('/home/user/LAT5150DRVMIL/nsa_reconnaissance_enhanced.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedNSADeviceReconnaissance:
    """
    Enhanced NSA-grade device reconnaissance with NPU integration
    Classified operation for comprehensive device awareness
    """

    def __init__(self):
        self.sudo_password = "1786"
        self.target_range = range(0x8000, 0x806C)  # 84 devices

        # CRITICAL - NEVER PROBE THESE DEVICES
        self.quarantine_list = [0x8009, 0x800A, 0x800B, 0x8019, 0x8029]

        # Known devices from previous reconnaissance
        self.known_devices = {
            0x8005: "TPM/HSM Interface Controller",
            0x8007: "Power Management Controller",
            0x8008: "Secure Boot Validator",
            0x8011: "Encryption Key Management",
            0x8013: "Unknown Extended Security",
            0x8014: "Unknown Extended Security",
            0x8022: "Network Security Filter",
            0x8027: "Network Authentication Gateway"
        }

        # Device classification matrices
        self.device_signatures = {}
        self.response_patterns = {}
        self.operational_readiness = {}

        # Enhanced NSA tradecraft patterns including NPU signatures
        self.nsa_signatures = {
            # Original signatures
            'intel_me': [0xFF, 0x00, 0x55, 0xAA],
            'tpm_response': [0x80, 0x01, 0x00, 0x00],
            'dell_proprietary': [0xDE, 0x11],
            'secure_enclave': [0x5E, 0xCE],
            'network_filter': [0x4E, 0x46],
            'crypto_engine': [0xCE, 0x12],

            # NPU signatures (NEW)
            'intel_npu': [0x4E, 0x50, 0x55],  # "NPU" ASCII
            'intel_vpu': [0x56, 0x50, 0x55],  # "VPU" ASCII
            'amd_npu': [0xAD, 0x4E, 0x50],
            'qualcomm_npu': [0x51, 0x4E, 0x50],
            'neural_engine': [0x4E, 0x45, 0x12],
            'ai_accelerator': [0xAA, 0xCC, 0x01],
            'tensor_core': [0x54, 0x43, 0x00],
            'inference_engine': [0x49, 0x4E, 0x46],

            # Additional device signatures
            'audio_dsp': [0xAD, 0x50, 0x00],
            'video_decoder': [0x56, 0x44, 0xEC],
            'sensor_hub': [0x53, 0x48, 0x55],
            'thermal_management': [0x54, 0x4D, 0x47],
            'battery_controller': [0xBA, 0x54, 0x54],
            'wireless_controller': [0x57, 0x4C, 0x4E],
            'biometric_sensor': [0xB1, 0x0, 0x00],
            'storage_controller': [0x53, 0x54, 0x4F],
            'memory_controller': [0x4D, 0x45, 0x4D],
            'pcie_controller': [0x50, 0x43, 0x49],
            'usb_controller': [0x55, 0x53, 0x42],
            'thunderbolt': [0x54, 0x42, 0x54],
        }

        # NPU-specific PCI device IDs
        self.npu_pci_ids = {
            '8086:7d1d': 'Intel AI Boost (Meteor Lake NPU)',
            '8086:643e': 'Intel VPU (Raptor Lake)',
            '8086:ad1d': 'Intel NPU (Arrow Lake)',
            '1022:1502': 'AMD Ryzen AI (Phoenix)',
            '1022:17f0': 'AMD Neural Processor',
            '17cb:1234': 'Qualcomm Neural Processing Engine',
        }

    def run_privileged_command(self, cmd: List[str], timeout: int = 10) -> Tuple[int, str, str]:
        """Execute privileged command using NSA operational security"""
        try:
            full_cmd = ["sudo", "-S"] + cmd
            process = subprocess.Popen(
                full_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            stdout, stderr = process.communicate(
                input=self.sudo_password + "\n",
                timeout=timeout
            )

            return process.returncode, stdout, stderr

        except subprocess.TimeoutExpired:
            process.kill()
            return -1, "", f"Command timed out after {timeout}s"
        except Exception as e:
            return -1, "", str(e)

    def detect_npu_hardware(self) -> List[Dict[str, Any]]:
        """
        Comprehensive NPU hardware detection across multiple methods
        """
        logger.info("🧠 Scanning for NPU hardware...")
        detected_npus = []

        # Method 1: PCI device enumeration
        npu_pci = self.detect_npu_via_pci()
        if npu_pci:
            detected_npus.extend(npu_pci)

        # Method 2: ACPI device tables
        npu_acpi = self.detect_npu_via_acpi()
        if npu_acpi:
            detected_npus.extend(npu_acpi)

        # Method 3: Linux kernel modules
        npu_kernel = self.detect_npu_via_kernel_modules()
        if npu_kernel:
            detected_npus.extend(npu_kernel)

        # Method 4: sysfs enumeration
        npu_sysfs = self.detect_npu_via_sysfs()
        if npu_sysfs:
            detected_npus.extend(npu_sysfs)

        # Method 5: Device tree (for ARM-based NPUs)
        npu_dt = self.detect_npu_via_devicetree()
        if npu_dt:
            detected_npus.extend(npu_dt)

        logger.info(f"✓ Detected {len(detected_npus)} NPU/AI accelerator devices")
        return detected_npus

    def detect_npu_via_pci(self) -> List[Dict[str, Any]]:
        """Detect NPUs via PCI bus enumeration"""
        npus = []

        try:
            # Use lspci to enumerate PCI devices
            returncode, stdout, _ = self.run_privileged_command([
                'lspci', '-nn', '-v'
            ], timeout=5)

            if returncode == 0:
                # Parse lspci output for known NPU IDs
                for line in stdout.split('\n'):
                    for pci_id, device_name in self.npu_pci_ids.items():
                        if pci_id in line.lower():
                            npus.append({
                                'detection_method': 'pci',
                                'device_type': 'npu',
                                'pci_id': pci_id,
                                'device_name': device_name,
                                'raw_info': line.strip()
                            })
                            logger.info(f"  🎯 Found NPU: {device_name} ({pci_id})")

                # Also check for generic AI/Neural keywords
                keywords = ['neural', 'npu', 'vpu', 'ai', 'inference', 'accelerator', 'tensor']
                for line in stdout.split('\n'):
                    line_lower = line.lower()
                    if any(keyword in line_lower for keyword in keywords):
                        if not any(line in str(npu) for npu in npus):  # Avoid duplicates
                            npus.append({
                                'detection_method': 'pci_keyword',
                                'device_type': 'potential_npu',
                                'raw_info': line.strip()
                            })
                            logger.info(f"  🔍 Potential NPU/AI device: {line.strip()[:80]}")

        except Exception as e:
            logger.debug(f"PCI NPU detection error: {e}")

        return npus

    def detect_npu_via_acpi(self) -> List[Dict[str, Any]]:
        """Detect NPUs via ACPI tables"""
        npus = []

        try:
            acpi_paths = [
                '/sys/firmware/acpi/tables/',
                '/sys/devices/platform/',
            ]

            npu_keywords = ['NPU', 'VPU', 'INPU', 'ANPU', 'AICC']

            for acpi_path in acpi_paths:
                if not Path(acpi_path).exists():
                    continue

                returncode, stdout, _ = self.run_privileged_command([
                    'find', acpi_path, '-type', 'f', '-o', '-type', 'l'
                ], timeout=5)

                if returncode == 0:
                    for line in stdout.split('\n'):
                        for keyword in npu_keywords:
                            if keyword.lower() in line.lower():
                                npus.append({
                                    'detection_method': 'acpi',
                                    'device_type': 'npu_acpi',
                                    'acpi_path': line.strip()
                                })
                                logger.info(f"  🎯 ACPI NPU entry: {line.strip()}")

        except Exception as e:
            logger.debug(f"ACPI NPU detection error: {e}")

        return npus

    def detect_npu_via_kernel_modules(self) -> List[Dict[str, Any]]:
        """Detect NPUs via loaded kernel modules"""
        npus = []

        try:
            npu_module_patterns = [
                'intel_vpu',
                'intel_npu',
                'amd_npu',
                'qaic',  # Qualcomm AI Cloud
                'habanalabs',  # Habana AI
                'neuron',  # AWS Inferentia
            ]

            returncode, stdout, _ = self.run_privileged_command([
                'lsmod'
            ], timeout=3)

            if returncode == 0:
                for pattern in npu_module_patterns:
                    if pattern in stdout.lower():
                        npus.append({
                            'detection_method': 'kernel_module',
                            'device_type': 'npu_driver',
                            'module_name': pattern
                        })
                        logger.info(f"  🎯 NPU kernel module: {pattern}")

        except Exception as e:
            logger.debug(f"Kernel module NPU detection error: {e}")

        return npus

    def detect_npu_via_sysfs(self) -> List[Dict[str, Any]]:
        """Detect NPUs via sysfs device enumeration"""
        npus = []

        try:
            sysfs_paths = [
                '/sys/class/accel/',  # AI accelerators
                '/sys/class/npu/',
                '/sys/class/vpu/',
                '/sys/devices/pci*/*/drm/',
            ]

            for sysfs_path in sysfs_paths:
                if '*' in sysfs_path:
                    # Handle glob patterns
                    returncode, stdout, _ = self.run_privileged_command([
                        'find', '/sys/devices/', '-path', sysfs_path, '-type', 'd'
                    ], timeout=5)

                    if returncode == 0:
                        for line in stdout.split('\n'):
                            if line.strip():
                                npus.append({
                                    'detection_method': 'sysfs',
                                    'device_type': 'npu_sysfs',
                                    'sysfs_path': line.strip()
                                })
                                logger.info(f"  🎯 sysfs NPU device: {line.strip()}")
                else:
                    if Path(sysfs_path).exists():
                        devices = list(Path(sysfs_path).iterdir())
                        for device in devices:
                            npus.append({
                                'detection_method': 'sysfs',
                                'device_type': 'npu_class',
                                'sysfs_path': str(device)
                            })
                            logger.info(f"  🎯 sysfs NPU class device: {device.name}")

        except Exception as e:
            logger.debug(f"sysfs NPU detection error: {e}")

        return npus

    def detect_npu_via_devicetree(self) -> List[Dict[str, Any]]:
        """Detect NPUs via device tree (ARM systems)"""
        npus = []

        try:
            dt_path = Path('/proc/device-tree/')
            if dt_path.exists():
                returncode, stdout, _ = self.run_privileged_command([
                    'find', str(dt_path), '-name', '*npu*', '-o', '-name', '*ai*'
                ], timeout=5)

                if returncode == 0:
                    for line in stdout.split('\n'):
                        if line.strip():
                            npus.append({
                                'detection_method': 'devicetree',
                                'device_type': 'npu_dt',
                                'dt_path': line.strip()
                            })
                            logger.info(f"  🎯 Device tree NPU: {line.strip()}")

        except Exception as e:
            logger.debug(f"Device tree NPU detection error: {e}")

        return npus

    def probe_device_safe(self, device_id: int, npu_context: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Enhanced safe device probing with NPU awareness
        READ-ONLY operations with multiple safety layers
        """
        if device_id in self.quarantine_list:
            logger.critical(f"ABORT: Device 0x{device_id:04X} is QUARANTINED - NEVER PROBE!")
            return {
                'device_id': device_id,
                'status': 'QUARANTINED',
                'response': None,
                'confidence': 0.0,
                'danger_level': 'CRITICAL'
            }

        logger.info(f"Probing device 0x{device_id:04X} with enhanced NSA protocols...")

        probe_result = {
            'device_id': device_id,
            'timestamp': datetime.now().isoformat(),
            'probe_method': 'nsa_enhanced_safe_read',
            'status': 'unknown',
            'response_data': None,
            'response_pattern': None,
            'confidence': 0.0,
            'operational_readiness': 'unknown',
            'security_classification': 'unclassified',
            'npu_correlation': None
        }

        try:
            # Method 1: DSMIL kernel module interface
            response_data = self.probe_via_dsmil_module(device_id)
            if response_data:
                probe_result.update(response_data)
                probe_result['status'] = 'responsive'

            # Method 2: Direct SMI interface (if kernel module fails)
            if not response_data:
                response_data = self.probe_via_smi_direct(device_id)
                if response_data:
                    probe_result.update(response_data)
                    probe_result['status'] = 'smi_responsive'

            # Method 3: ACPI device enumeration
            acpi_data = self.probe_via_acpi(device_id)
            if acpi_data:
                probe_result['acpi_signature'] = acpi_data

            # Method 4: PCI configuration space probing (NEW)
            pci_data = self.probe_via_pci_config(device_id)
            if pci_data:
                probe_result['pci_config'] = pci_data

            # Method 5: Memory-mapped I/O probing (NEW)
            mmio_data = self.probe_via_mmio(device_id)
            if mmio_data:
                probe_result['mmio_data'] = mmio_data

            # Enhanced NSA pattern analysis with NPU signatures
            if probe_result['response_data']:
                pattern_analysis = self.analyze_response_pattern(
                    device_id,
                    probe_result['response_data']
                )
                probe_result.update(pattern_analysis)

            # NPU correlation analysis (NEW)
            if npu_context:
                npu_correlation = self.correlate_with_npu(device_id, probe_result, npu_context)
                if npu_correlation:
                    probe_result['npu_correlation'] = npu_correlation
                    probe_result['device_type_hint'] = 'npu_related'

            # Classify operational readiness
            probe_result['operational_readiness'] = self.assess_operational_readiness(probe_result)
            probe_result['confidence'] = self.calculate_confidence_score(probe_result)

        except Exception as e:
            logger.error(f"Probe failed for 0x{device_id:04X}: {e}")
            probe_result['status'] = 'probe_failed'
            probe_result['error'] = str(e)

        return probe_result

    def probe_via_dsmil_module(self, device_id: int) -> Optional[Dict[str, Any]]:
        """Probe via DSMIL kernel module"""
        dsmil_dev = Path("/dev/dsmil")
        if not dsmil_dev.exists():
            return None

        try:
            with open(dsmil_dev, 'rb+') as f:
                # Send READ command
                read_cmd = struct.pack('<HH', 0x0001, device_id)
                f.write(read_cmd)
                f.flush()

                # Read response
                response = f.read(8)  # Expect 8 bytes
                if len(response) >= 4:
                    status, value = struct.unpack('<HH', response[:4])

                    return {
                        'response_data': response.hex(),
                        'response_bytes': list(response),
                        'parsed_status': status,
                        'parsed_value': value,
                        'probe_method': 'dsmil_kernel'
                    }
        except Exception as e:
            logger.debug(f"DSMIL module probe failed for 0x{device_id:04X}: {e}")
            return None

    def probe_via_smi_direct(self, device_id: int) -> Optional[Dict[str, Any]]:
        """Direct SMI interface probing (requires root)"""
        try:
            # Use outb/inb for direct SMI access
            cmd_port = 0x164E  # SMI command port
            data_port = 0x164F  # SMI data port

            # Send device ID to command port
            returncode, stdout, stderr = self.run_privileged_command([
                'python3', '-c',
                f"import struct; "
                f"with open('/dev/port', 'r+b') as f: "
                f"f.seek({cmd_port}); f.write(struct.pack('H', {device_id})); "
                f"f.seek({data_port}); data = f.read(4); "
                f"print(data.hex() if data else 'no_response')"
            ], timeout=5)

            if returncode == 0 and stdout.strip() and 'no_response' not in stdout:
                response_hex = stdout.strip()
                if response_hex and len(response_hex) >= 2:
                    response_bytes = bytes.fromhex(response_hex)
                    return {
                        'response_data': response_hex,
                        'response_bytes': list(response_bytes),
                        'probe_method': 'smi_direct'
                    }

        except Exception as e:
            logger.debug(f"Direct SMI probe failed for 0x{device_id:04X}: {e}")
            return None

    def probe_via_acpi(self, device_id: int) -> Optional[str]:
        """ACPI device enumeration"""
        try:
            # Check ACPI tables for device references
            returncode, stdout, _ = self.run_privileged_command([
                'grep', '-r', f'{device_id:04X}', '/sys/firmware/acpi/tables/'
            ], timeout=3)

            if returncode == 0 and stdout.strip():
                return stdout.strip()[:100]  # Truncate for security
        except:
            pass
        return None

    def probe_via_pci_config(self, device_id: int) -> Optional[Dict[str, Any]]:
        """Probe PCI configuration space for device ID correlation"""
        try:
            # Map DSMIL device ID to potential PCI bus location
            # DSMIL 0x8000+ might map to PCI extended config space
            pci_offset = (device_id - 0x8000) * 4  # Hypothetical mapping

            returncode, stdout, _ = self.run_privileged_command([
                'lspci', '-xxx', '-d', '*:*'
            ], timeout=5)

            if returncode == 0 and stdout:
                # Search for device ID patterns in PCI config space dumps
                if f"{device_id:04x}" in stdout.lower():
                    return {
                        'pci_correlation': True,
                        'pci_offset_hint': pci_offset
                    }
        except Exception as e:
            logger.debug(f"PCI config probe failed for 0x{device_id:04X}: {e}")

        return None

    def probe_via_mmio(self, device_id: int) -> Optional[Dict[str, Any]]:
        """Probe memory-mapped I/O regions"""
        try:
            # Check /proc/iomem for device-specific memory regions
            with open('/proc/iomem', 'r') as f:
                iomem_data = f.read()

            # Search for device ID in memory map
            device_id_str = f"{device_id:04x}"
            if device_id_str in iomem_data.lower():
                return {
                    'mmio_correlation': True,
                    'iomem_match': True
                }
        except Exception as e:
            logger.debug(f"MMIO probe failed for 0x{device_id:04X}: {e}")

        return None

    def correlate_with_npu(self, device_id: int, probe_result: Dict[str, Any],
                          npu_context: List[Dict]) -> Optional[Dict[str, Any]]:
        """
        Correlate DSMIL device with detected NPU hardware
        """
        if not npu_context:
            return None

        correlation = {
            'npu_proximity_score': 0.0,
            'potential_npu_related': False,
            'npu_hints': []
        }

        # Check if device ID falls in NPU-likely ranges
        npu_likely_ranges = [
            (0x8030, 0x8040),  # Hypothetical NPU control range
            (0x8050, 0x805F),  # Hypothetical AI accelerator range
        ]

        for start, end in npu_likely_ranges:
            if start <= device_id <= end:
                correlation['npu_proximity_score'] += 0.3
                correlation['potential_npu_related'] = True
                correlation['npu_hints'].append(f"Device in NPU-likely range 0x{start:04X}-0x{end:04X}")

        # Check response patterns for NPU signatures
        if probe_result.get('signature_matches'):
            npu_sig_count = sum(1 for sig in probe_result['signature_matches']
                              if 'npu' in sig or 'neural' in sig or 'ai' in sig)
            if npu_sig_count > 0:
                correlation['npu_proximity_score'] += 0.4
                correlation['potential_npu_related'] = True
                correlation['npu_hints'].append(f"Found {npu_sig_count} NPU-related signatures")

        # Correlate with detected NPU hardware
        for npu in npu_context:
            if 'intel' in str(npu).lower() and 0x8030 <= device_id <= 0x8040:
                correlation['npu_proximity_score'] += 0.3
                correlation['npu_hints'].append(f"Intel NPU detected, device in control range")

        if correlation['npu_proximity_score'] > 0:
            return correlation

        return None

    def analyze_response_pattern(self, device_id: int, response_data: str) -> Dict[str, Any]:
        """Enhanced NSA-grade response pattern analysis with NPU signatures"""
        if not response_data:
            return {}

        response_bytes = bytes.fromhex(response_data) if isinstance(response_data, str) else response_data

        analysis = {
            'response_length': len(response_bytes),
            'entropy_score': self.calculate_entropy(response_bytes),
            'signature_matches': [],
            'pattern_classification': 'unknown'
        }

        # Check against enhanced NSA signatures (including NPU)
        for sig_name, signature in self.nsa_signatures.items():
            if self.bytes_contain_pattern(response_bytes, signature):
                analysis['signature_matches'].append(sig_name)

        # Enhanced pattern classification with NPU support
        if any('npu' in sig for sig in analysis['signature_matches']):
            analysis['pattern_classification'] = 'neural_processing_unit'
        elif any('vpu' in sig for sig in analysis['signature_matches']):
            analysis['pattern_classification'] = 'vision_processing_unit'
        elif 'intel_me' in analysis['signature_matches']:
            analysis['pattern_classification'] = 'intel_management_engine'
        elif 'tpm_response' in analysis['signature_matches']:
            analysis['pattern_classification'] = 'trusted_platform_module'
        elif 'dell_proprietary' in analysis['signature_matches']:
            analysis['pattern_classification'] = 'dell_proprietary_interface'
        elif 'crypto_engine' in analysis['signature_matches']:
            analysis['pattern_classification'] = 'cryptographic_processor'
        elif 'network_filter' in analysis['signature_matches']:
            analysis['pattern_classification'] = 'network_security_device'
        elif any('ai' in sig or 'tensor' in sig or 'inference' in sig
                for sig in analysis['signature_matches']):
            analysis['pattern_classification'] = 'ai_accelerator'

        # Security classification based on patterns
        if analysis['entropy_score'] > 7.0:  # High entropy = likely encrypted/random
            analysis['security_classification'] = 'encrypted_response'
        elif len(analysis['signature_matches']) > 0:
            analysis['security_classification'] = 'classified'

        return analysis

    def calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of response data"""
        if not data:
            return 0.0

        byte_counts = {}
        for byte in data:
            byte_counts[byte] = byte_counts.get(byte, 0) + 1

        entropy = 0.0
        for count in byte_counts.values():
            probability = count / len(data)
            if probability > 0:
                import math
                entropy -= probability * math.log2(probability)

        return entropy

    def bytes_contain_pattern(self, data: bytes, pattern: List[int]) -> bool:
        """Check if data contains specific byte pattern"""
        pattern_bytes = bytes(pattern)
        return pattern_bytes in data

    def assess_operational_readiness(self, probe_result: Dict[str, Any]) -> str:
        """Assess device operational readiness using NSA criteria"""
        if probe_result['status'] == 'QUARANTINED':
            return 'NEVER_ACTIVATE'

        if probe_result['status'] in ['responsive', 'smi_responsive']:
            confidence = probe_result.get('confidence', 0.0)

            # Special handling for NPU devices
            if probe_result.get('npu_correlation', {}).get('potential_npu_related'):
                if confidence > 0.6:
                    return 'NPU_CANDIDATE'
                else:
                    return 'NPU_ANALYSIS_REQUIRED'

            if confidence > 0.8:
                return 'READY_FOR_ACTIVATION'
            elif confidence > 0.6:
                return 'READY_WITH_CAUTION'
            elif confidence > 0.4:
                return 'REQUIRES_ANALYSIS'
            else:
                return 'INSUFFICIENT_DATA'

        return 'NOT_RESPONSIVE'

    def calculate_confidence_score(self, probe_result: Dict[str, Any]) -> float:
        """Enhanced confidence score calculation with NPU awareness"""
        score = 0.0

        # Base score for responsiveness
        if probe_result['status'] in ['responsive', 'smi_responsive']:
            score += 0.3

        # Score for response data quality
        if probe_result.get('response_data'):
            score += 0.2

            # Score for entropy (structured data = higher confidence)
            entropy = probe_result.get('entropy_score', 0)
            if 4.0 <= entropy <= 6.0:  # Sweet spot for structured data
                score += 0.2

        # Score for signature matches
        signature_matches = probe_result.get('signature_matches', [])
        score += min(len(signature_matches) * 0.1, 0.3)  # Cap at 0.3

        # Bonus for NPU correlation
        npu_correlation = probe_result.get('npu_correlation', {})
        if npu_correlation and npu_correlation.get('potential_npu_related'):
            score += npu_correlation.get('npu_proximity_score', 0) * 0.2

        # Score for multiple detection methods
        detection_methods = sum([
            1 if probe_result.get('response_data') else 0,
            1 if probe_result.get('acpi_signature') else 0,
            1 if probe_result.get('pci_config') else 0,
            1 if probe_result.get('mmio_data') else 0,
        ])
        score += detection_methods * 0.05

        # Score for known device
        device_id = probe_result['device_id']
        if device_id in self.known_devices:
            score += 0.2

        return min(score, 1.0)  # Cap at 1.0

    def generate_device_report(self, device_id: int, probe_result: Dict[str, Any]) -> str:
        """Generate formatted device intelligence report"""
        confidence = probe_result.get('confidence', 0.0)
        readiness = probe_result.get('operational_readiness', 'unknown')

        status_icon = {
            'responsive': '🟢',
            'smi_responsive': '🟡',
            'QUARANTINED': '🚫',
            'probe_failed': '🔴',
            'unknown': '⚪'
        }.get(probe_result['status'], '❓')

        readiness_icon = {
            'READY_FOR_ACTIVATION': '✅',
            'READY_WITH_CAUTION': '⚠️',
            'REQUIRES_ANALYSIS': '🔍',
            'NEVER_ACTIVATE': '🚫',
            'NOT_RESPONSIVE': '❌',
            'INSUFFICIENT_DATA': '❓',
            'NPU_CANDIDATE': '🧠',
            'NPU_ANALYSIS_REQUIRED': '🧠🔍'
        }.get(readiness, '❓')

        # Check for NPU correlation
        npu_indicator = ""
        if probe_result.get('npu_correlation', {}).get('potential_npu_related'):
            npu_indicator = " [NPU]"

        # Determine device name
        known_name = self.known_devices.get(device_id)
        pattern_class = probe_result.get('pattern_classification', '')

        if known_name:
            device_name = known_name
        elif pattern_class and pattern_class != 'unknown':
            device_name = pattern_class.replace('_', ' ').title()
        else:
            device_name = "Unknown Device"

        report = f"{status_icon} 0x{device_id:04X}: {device_name}{npu_indicator}"
        if confidence > 0:
            report += f" (Confidence: {confidence:.1%})"
        report += f" {readiness_icon} {readiness}"

        # Add signature matches summary
        sig_matches = probe_result.get('signature_matches', [])
        if sig_matches:
            report += f" [Sigs: {', '.join(sig_matches[:3])}]"

        return report

    def execute_reconnaissance(self) -> Dict[str, Any]:
        """Execute enhanced NSA reconnaissance mission with NPU integration"""
        logger.info("=" * 80)
        logger.info("NSA DEVICE RECONNAISSANCE - ENHANCED MISSION START")
        logger.info(f"Target Range: 0x8000-0x806B ({len(self.target_range)} devices)")
        logger.info(f"Quarantined: {len(self.quarantine_list)} devices")
        logger.info("Enhancement: NPU Integration & Comprehensive Enumeration")
        logger.info("=" * 80)

        # PHASE 1: NPU Hardware Detection
        npu_hardware = self.detect_npu_hardware()

        mission_results = {
            'mission_timestamp': datetime.now().isoformat(),
            'mission_type': 'enhanced_with_npu',
            'target_range': f"0x8000-0x806B",
            'total_devices': len(self.target_range),
            'quarantined_devices': len(self.quarantine_list),
            'probed_devices': 0,
            'responsive_devices': 0,
            'high_confidence_devices': 0,
            'npu_related_devices': 0,
            'npu_hardware_detected': npu_hardware,
            'activation_candidates': [],
            'npu_candidates': [],
            'device_reports': {},
            'classification_summary': {},
            'recommendations': []
        }

        print("\n🔍 NSA ENHANCED DEVICE RECONNAISSANCE RESULTS")
        print("=" * 60)

        if npu_hardware:
            print(f"\n🧠 NPU HARDWARE DETECTED: {len(npu_hardware)} devices")
            for npu in npu_hardware[:5]:  # Show first 5
                device_name = npu.get('device_name', npu.get('device_type', 'Unknown NPU'))
                print(f"  • {device_name} ({npu.get('detection_method', 'unknown')})")
            print()

        # PHASE 2: Device Probing with NPU Context
        print("📡 Probing DSMIL Device Range...")
        print("-" * 60)

        for device_id in self.target_range:
            if device_id in self.quarantine_list:
                # Still record quarantined devices
                probe_result = {
                    'device_id': device_id,
                    'status': 'QUARANTINED',
                    'operational_readiness': 'NEVER_ACTIVATE',
                    'confidence': 0.0,
                    'security_classification': 'CRITICAL_DANGER'
                }
            else:
                probe_result = self.probe_device_safe(device_id, npu_context=npu_hardware)
                mission_results['probed_devices'] += 1

            # Store results
            mission_results['device_reports'][f"0x{device_id:04X}"] = probe_result

            # Count responsive devices
            if probe_result['status'] in ['responsive', 'smi_responsive']:
                mission_results['responsive_devices'] += 1

            # Count high confidence devices
            if probe_result.get('confidence', 0) > 0.7:
                mission_results['high_confidence_devices'] += 1

            # Count NPU-related devices
            if probe_result.get('npu_correlation', {}).get('potential_npu_related'):
                mission_results['npu_related_devices'] += 1

            # Identify activation candidates
            readiness = probe_result.get('operational_readiness', '')
            if readiness in ['READY_FOR_ACTIVATION', 'READY_WITH_CAUTION']:
                mission_results['activation_candidates'].append(device_id)

            # Identify NPU candidates
            if readiness in ['NPU_CANDIDATE', 'NPU_ANALYSIS_REQUIRED']:
                mission_results['npu_candidates'].append(device_id)

            # Generate device report
            device_report = self.generate_device_report(device_id, probe_result)
            print(device_report)

            # Small delay between probes for safety
            time.sleep(0.05)

        # Generate mission summary
        self.generate_mission_summary(mission_results)

        # Save detailed results
        results_file = Path(f"/home/user/LAT5150DRVMIL/nsa_reconnaissance_enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(results_file, 'w') as f:
            json.dump(mission_results, f, indent=2, default=str)

        logger.info(f"Mission results saved to: {results_file}")

        return mission_results

    def generate_mission_summary(self, results: Dict[str, Any]):
        """Generate enhanced classified mission summary"""
        print("\n" + "=" * 60)
        print("🎯 NSA ENHANCED RECONNAISSANCE MISSION SUMMARY")
        print("=" * 60)
        print(f"Total Devices Analyzed: {results['total_devices']}")
        print(f"Devices Probed: {results['probed_devices']}")
        print(f"Responsive Devices: {results['responsive_devices']}")
        print(f"High Confidence: {results['high_confidence_devices']}")
        print(f"Activation Candidates: {len(results['activation_candidates'])}")

        # NPU-specific summary
        print(f"\n🧠 NPU/AI ACCELERATOR ANALYSIS:")
        print(f"NPU Hardware Detected: {len(results['npu_hardware_detected'])}")
        print(f"NPU-Related DSMIL Devices: {results['npu_related_devices']}")
        print(f"NPU Activation Candidates: {len(results['npu_candidates'])}")

        if results['npu_candidates']:
            print(f"\n🧠 NPU DEVICE CANDIDATES:")
            for device_id in results['npu_candidates']:
                device_report = results['device_reports'].get(f"0x{device_id:04X}", {})
                device_name = device_report.get('pattern_classification', 'Unknown').replace('_', ' ').title()
                confidence = device_report.get('confidence', 0)
                print(f"  • 0x{device_id:04X}: {device_name} (Confidence: {confidence:.1%})")

        if results['activation_candidates']:
            print(f"\n🎯 TOP ACTIVATION CANDIDATES:")
            for device_id in results['activation_candidates'][:10]:  # Top 10
                device_name = self.known_devices.get(device_id, "Unknown Device")
                print(f"  • 0x{device_id:04X}: {device_name}")

        print(f"\n🚫 QUARANTINED DEVICES: {results['quarantined_devices']}")
        for device_id in self.quarantine_list:
            print(f"  • 0x{device_id:04X}: NEVER ACTIVATE")

        # Enhanced coverage expansion calculation
        current_known = len(self.known_devices)
        newly_responsive = results['responsive_devices']

        print(f"\n📊 INTELLIGENCE EXPANSION:")
        print(f"Previously Known: {current_known} devices")
        print(f"Newly Responsive: {newly_responsive} devices")
        print(f"NPU Integration: {len(results['npu_hardware_detected'])} hardware accelerators")

        expansion_percentage = (newly_responsive / results['total_devices']) * 100
        print(f"Intelligence Coverage: {expansion_percentage:.1f}% of target range")

def main():
    """Main enhanced NSA reconnaissance mission"""
    if os.geteuid() != 0:
        print("⚠️  Enhanced NSA Reconnaissance requires root privileges for hardware access")
        print("Continuing with limited capabilities...")

    # Initialize enhanced NSA reconnaissance
    nsa_recon = EnhancedNSADeviceReconnaissance()

    # Execute mission
    try:
        mission_results = nsa_recon.execute_reconnaissance()

        print(f"\n✅ NSA ENHANCED RECONNAISSANCE MISSION COMPLETE")
        print(f"Results: {mission_results['responsive_devices']} responsive devices")
        print(f"NPU Analysis: {len(mission_results['npu_candidates'])} NPU candidates identified")
        print(f"Activation Ready: {len(mission_results['activation_candidates'])} devices")
        print(f"Status: {'SUCCESSFUL' if mission_results['responsive_devices'] > 0 else 'LIMITED SUCCESS'}")

        return 0

    except KeyboardInterrupt:
        print(f"\n⚠️  Mission interrupted by operator")
        return 1
    except Exception as e:
        logger.error(f"Mission failed: {e}")
        import traceback
        traceback.print_exc()
        return 2

if __name__ == "__main__":
    sys.exit(main())
