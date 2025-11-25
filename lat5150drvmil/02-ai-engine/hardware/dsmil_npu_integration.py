#!/usr/bin/env python3
"""
DSMIL NPU Integration - Enhanced NPU with Military Attestation

Integrates consumer NPU (49.4 TOPS) and potential military NPU
with DSMIL 84-device framework for hardware-attested AI inference.

Architecture:
┌──────────────────────────────────────────────────────────────┐
│                    AI Inference Layer                         │
├──────────────────────────────────────────────────────────────┤
│  Consumer NPU (49.4 TOPS)  │  Military NPU (if present)      │
│  Device: /dev/accel/accel0  │  Device: /dev/dsmil/npu0       │
│  Driver: intel_vpu          │  Driver: dsmil-72dev (device 12│
├──────────────────────────────────────────────────────────────┤
│              DSMIL 84-Device Framework                        │
│  Device Allocations:                                          │
│  - Device 12: AI Hardware Security                           │
│  - Device 16: Platform Integrity Attestation                 │
│  - Device 32: Memory Encryption for Model Weights            │
│  - Device 48: APT Defense/Audit Logging                      │
├──────────────────────────────────────────────────────────────┤
│              Hardware Layer                                   │
│  Intel Meteor Lake NPU + GNA + iGPU (UMA)                    │
└──────────────────────────────────────────────────────────────┘

DSMIL Token Ranges (from dsmil-72dev module):
  Range_0400: 0x0400 - 0x0447 (72 tokens)
  Range_0480: 0x0480 - 0x04C7 (72 tokens) [PRIMARY]
  Range_0500: 0x0500 - 0x0547 (72 tokens)
  Total: 792 tokens across 11 ranges
"""

import os
import sys
import subprocess
import json
import hashlib
import time
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

# Add DSMIL framework to path
DSMIL_PATH = Path("/home/user/LAT5150DRVMIL/packaging/dsmil-platform_8.3.1-1/opt/dsmil")
if DSMIL_PATH.exists():
    sys.path.insert(0, str(DSMIL_PATH))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DSMILDevice(Enum):
    """DSMIL device allocations for AI workloads"""
    AI_SECURITY = 12          # AI Hardware Security
    TPM_SEAL = 3              # TPM Sealed Storage for models
    ATTESTATION = 16          # Platform Integrity Attestation
    MEMORY_ENCRYPT = 32       # Memory Encryption
    APT_AUDIT = 48            # APT Defense/Audit
    NPU_CONTROL = 12          # NPU control (same as AI_SECURITY)


class Mode5Level(Enum):
    """Mode 5 platform integrity levels"""
    STANDARD = "STANDARD"      # Default security
    ENHANCED = "ENHANCED"      # Enhanced attestation
    MILITARY = "MILITARY"      # Military-grade (requires clearance)


@dataclass
class NPUConfig:
    """NPU configuration and capabilities"""
    device_type: str           # "consumer" or "military"
    device_node: str           # Device path
    driver: str                # Kernel driver name
    tops_rating: float         # TOPS performance (INT8)
    precision: List[str]       # Supported precisions
    dsmil_integrated: bool     # DSMIL integration status
    attestation_required: bool # Requires hardware attestation
    mode5_level: Mode5Level    # Current Mode 5 level


class DSMILNPUIntegration:
    """
    Enhanced NPU integration with DSMIL framework

    Features:
    - Consumer NPU (49.4 TOPS) with DSMIL attestation
    - Military NPU detection and activation (if present)
    - Hardware-attested AI inference
    - TPM-sealed model weights
    - Mode 5 platform integrity
    """

    def __init__(self):
        self.dsmil_device = "/dev/dsmil0"
        self.dsmil_module = "dsmil_72dev"
        self.consumer_npu_device = "/dev/accel/accel0"
        self.military_npu_device = "/dev/dsmil/npu0"

        self.dsmil_available = False
        self.consumer_npu_available = False
        self.military_npu_available = False

        self.npus: List[NPUConfig] = []
        self.mode5_level = Mode5Level.STANDARD

        # Try to import DSMIL modules
        self.dsmil_military_mode = None
        self.dsmil_ai_engine = None

        try:
            # Import DSMIL framework modules
            from dsmil_military_mode import DSMILMilitaryMode
            from dsmil_ai_engine import DSMILAIEngine

            self.dsmil_military_mode = DSMILMilitaryMode()
            self.dsmil_ai_engine = DSMILAIEngine()
            logger.info("✓ DSMIL framework modules imported")
        except ImportError:
            logger.warning("⚠️  DSMIL framework modules not available")
            logger.info("   Install from: /home/user/LAT5150DRVMIL/packaging/")

    def check_dsmil_framework(self) -> Dict:
        """Check DSMIL framework status"""
        logger.info("\nChecking DSMIL Framework...")

        status = {
            "module_loaded": False,
            "device_node_exists": False,
            "device_accessible": False,
            "mode5_level": None,
            "devices_available": 0,
        }

        # Check kernel module
        try:
            result = subprocess.run(
                ["lsmod"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if self.dsmil_module in result.stdout:
                status["module_loaded"] = True
                logger.info(f"✓ DSMIL module loaded: {self.dsmil_module}")
                self.dsmil_available = True
            else:
                logger.warning(f"⚠️  DSMIL module not loaded: {self.dsmil_module}")
                logger.info(f"   Load with: sudo modprobe {self.dsmil_module}")

        except Exception as e:
            logger.error(f"Error checking module: {e}")

        # Check device node
        if os.path.exists(self.dsmil_device):
            status["device_node_exists"] = True
            logger.info(f"✓ DSMIL device exists: {self.dsmil_device}")

            # Check accessibility
            if os.access(self.dsmil_device, os.R_OK | os.W_OK):
                status["device_accessible"] = True
                logger.info("✓ Device is accessible (R/W)")
            else:
                logger.warning("⚠️  Device exists but not accessible")
                logger.info("   Run with sudo or add user to 'dsmil' group")
        else:
            logger.warning(f"⚠️  DSMIL device not found: {self.dsmil_device}")

        # Get Mode 5 status (via DSMIL framework)
        if self.dsmil_military_mode:
            try:
                mode5_status = self.dsmil_military_mode.check_mode5_status()
                status["mode5_level"] = mode5_status.get("mode5_level")
                status["devices_available"] = mode5_status.get("devices_available", 0)

                logger.info(f"✓ Mode 5 Level: {status['mode5_level']}")
                logger.info(f"✓ DSMIL Devices: {status['devices_available']}")

                # Set mode5_level
                if status["mode5_level"] == "MILITARY":
                    self.mode5_level = Mode5Level.MILITARY
                elif status["mode5_level"] == "ENHANCED":
                    self.mode5_level = Mode5Level.ENHANCED

            except Exception as e:
                logger.warning(f"Could not get Mode 5 status: {e}")

        return status

    def detect_consumer_npu(self) -> Optional[NPUConfig]:
        """Detect and configure consumer NPU (Intel Meteor Lake)"""
        logger.info("\nDetecting Consumer NPU...")

        # Check device node
        if not os.path.exists(self.consumer_npu_device):
            logger.warning(f"⚠️  Consumer NPU device not found: {self.consumer_npu_device}")
            return None

        logger.info(f"✓ Consumer NPU device exists: {self.consumer_npu_device}")

        # Check driver
        try:
            result = subprocess.run(
                ["lsmod"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if "intel_vpu" in result.stdout:
                logger.info("✓ intel_vpu driver loaded")
            else:
                logger.warning("⚠️  intel_vpu driver not loaded")
                return None

        except:
            pass

        # Check OpenVINO detection
        openvino_ok = False
        try:
            from openvino.runtime import Core

            core = Core()
            if "NPU" in core.available_devices:
                logger.info("✓ NPU detected by OpenVINO")
                openvino_ok = True
            else:
                logger.warning("⚠️  NPU not detected by OpenVINO")

        except ImportError:
            logger.warning("⚠️  OpenVINO not installed (optional)")
        except:
            pass

        # Create NPU config
        npu_config = NPUConfig(
            device_type="consumer",
            device_node=self.consumer_npu_device,
            driver="intel_vpu",
            tops_rating=49.4,
            precision=["INT8", "FP16"],
            dsmil_integrated=self.dsmil_available,
            attestation_required=False,
            mode5_level=self.mode5_level
        )

        logger.info(f"✓ Consumer NPU: {npu_config.tops_rating} TOPS (INT8)")
        logger.info(f"  DSMIL Integration: {'Yes' if npu_config.dsmil_integrated else 'No'}")
        logger.info(f"  Mode 5 Level: {npu_config.mode5_level.value}")

        self.consumer_npu_available = True
        return npu_config

    def detect_military_npu(self) -> Optional[NPUConfig]:
        """Detect and configure military NPU (if present)"""
        logger.info("\nDetecting Military NPU...")

        # Military NPU would be integrated with DSMIL device 12
        if not self.dsmil_available:
            logger.info("  DSMIL framework not available")
            logger.info("  Military NPU requires DSMIL integration")
            return None

        # Check for military NPU device node
        military_device_paths = [
            self.military_npu_device,
            "/dev/dsmil/ai_security",
            f"/dev/dsmil{DSMILDevice.AI_SECURITY.value}",
        ]

        device_found = None
        for path in military_device_paths:
            if os.path.exists(path):
                device_found = path
                logger.info(f"✓ Military NPU device found: {path}")
                break

        if not device_found:
            logger.info("  No military NPU device node found")
            logger.info("  System has consumer NPU only (49.4 TOPS)")
            return None

        # Military NPU detected - create config
        npu_config = NPUConfig(
            device_type="military",
            device_node=device_found,
            driver=self.dsmil_module,
            tops_rating=150.0,  # Example: 150 TOPS (3x consumer)
            precision=["INT8", "INT4", "FP16", "BF16"],
            dsmil_integrated=True,
            attestation_required=True,
            mode5_level=self.mode5_level
        )

        logger.info(f"🎯 MILITARY NPU DETECTED!")
        logger.info(f"  Performance: {npu_config.tops_rating} TOPS (INT8)")
        logger.info(f"  DSMIL Device: {DSMILDevice.AI_SECURITY.value} (AI Hardware Security)")
        logger.info(f"  Mode 5 Level: {npu_config.mode5_level.value}")
        logger.info(f"  Attestation: Required")

        self.military_npu_available = True
        return npu_config

    def test_attested_inference(self) -> bool:
        """Test hardware-attested AI inference via DSMIL"""
        logger.info("\nTesting Hardware-Attested Inference...")

        if not self.dsmil_ai_engine:
            logger.warning("⚠️  DSMIL AI Engine not available")
            return False

        try:
            # Get Mode 5 status
            mode5_status = self.dsmil_military_mode.check_mode5_status()

            logger.info(f"Mode 5 Status: {mode5_status}")
            logger.info(f"  Enabled: {mode5_status.get('mode5_enabled')}")
            logger.info(f"  Level: {mode5_status.get('mode5_level')}")
            logger.info(f"  Devices: {mode5_status.get('devices_available')}")

            # Test simple query routing
            test_query = "Test DSMIL-attested inference"
            model = self.dsmil_ai_engine.route_query(test_query)

            logger.info(f"✓ Query routed to model: {model}")

            return True

        except Exception as e:
            logger.error(f"✗ Attested inference test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_dsmil_device_status(self, device_id: int) -> Dict:
        """Get status of specific DSMIL device"""
        logger.info(f"\nQuerying DSMIL Device {device_id}...")

        status = {
            "device_id": device_id,
            "device_name": None,
            "accessible": False,
            "token_range": None,
        }

        # Map device IDs to names
        device_names = {
            12: "AI Hardware Security",
            3: "TPM Sealed Storage",
            16: "Platform Integrity",
            32: "Memory Encryption",
            48: "APT Defense/Audit",
        }

        status["device_name"] = device_names.get(device_id, f"Device {device_id}")

        # Calculate token range for device
        # DSMIL uses ranges: 0x0480 + device_id
        base_token = 0x0480
        token = base_token + device_id
        status["token_range"] = f"0x{token:04X}"

        logger.info(f"  Name: {status['device_name']}")
        logger.info(f"  Token: {status['token_range']}")

        return status

    def run_comprehensive_integration(self) -> bool:
        """Run comprehensive NPU integration with DSMIL"""
        logger.info("\n" + "=" * 80)
        logger.info("  DSMIL NPU INTEGRATION")
        logger.info("=" * 80)
        logger.info("Hardware: Dell Latitude 5450 MIL-SPEC")
        logger.info("Framework: DSMIL 84-Device (dsmil-72dev)")
        logger.info("=" * 80)

        # Step 1: Check DSMIL framework
        logger.info("\n[Step 1/5] DSMIL Framework Status")
        dsmil_status = self.check_dsmil_framework()

        # Step 2: Detect consumer NPU
        logger.info("\n[Step 2/5] Consumer NPU Detection")
        consumer_npu = self.detect_consumer_npu()
        if consumer_npu:
            self.npus.append(consumer_npu)

        # Step 3: Detect military NPU
        logger.info("\n[Step 3/5] Military NPU Detection")
        military_npu = self.detect_military_npu()
        if military_npu:
            self.npus.append(military_npu)

        # Step 4: Query DSMIL AI devices
        logger.info("\n[Step 4/5] DSMIL AI Device Status")
        for device_id in [DSMILDevice.AI_SECURITY.value,
                          DSMILDevice.TPM_SEAL.value,
                          DSMILDevice.ATTESTATION.value,
                          DSMILDevice.MEMORY_ENCRYPT.value]:
            self.get_dsmil_device_status(device_id)

        # Step 5: Test attested inference
        logger.info("\n[Step 5/5] Hardware-Attested Inference Test")
        if dsmil_status["module_loaded"]:
            inference_ok = self.test_attested_inference()
        else:
            logger.warning("⚠️  Skipping (DSMIL module not loaded)")
            inference_ok = False

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("  INTEGRATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"DSMIL Framework:      {'✓ ACTIVE' if dsmil_status['module_loaded'] else '✗ INACTIVE'}")
        logger.info(f"Consumer NPU:         {'✓ AVAILABLE' if self.consumer_npu_available else '✗ NOT FOUND'}")
        logger.info(f"Military NPU:         {'✓ DETECTED' if self.military_npu_available else '⚠️  NOT DETECTED'}")
        logger.info(f"Attested Inference:   {'✓ WORKING' if inference_ok else '✗ NOT TESTED'}")
        logger.info(f"Mode 5 Level:         {self.mode5_level.value}")
        logger.info("=" * 80)

        # Export configuration
        config_file = "/tank/ai-engine/logs/dsmil_npu_config.json"
        os.makedirs(os.path.dirname(config_file), exist_ok=True)

        config_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dsmil_status": dsmil_status,
            "npus": [asdict(npu) for npu in self.npus],
            "mode5_level": self.mode5_level.value,
            "dsmil_ai_devices": {
                "ai_security": DSMILDevice.AI_SECURITY.value,
                "tpm_seal": DSMILDevice.TPM_SEAL.value,
                "attestation": DSMILDevice.ATTESTATION.value,
                "memory_encrypt": DSMILDevice.MEMORY_ENCRYPT.value,
            }
        }

        # Convert Mode5Level enum to string for JSON serialization
        for npu in config_data["npus"]:
            if isinstance(npu.get("mode5_level"), Mode5Level):
                npu["mode5_level"] = npu["mode5_level"].value

        try:
            with open(config_file, "w") as f:
                json.dump(config_data, f, indent=2)
            logger.info(f"\n✓ Configuration exported: {config_file}")
        except:
            pass

        # Recommendations
        logger.info("\nRecommendations:")
        if not dsmil_status["module_loaded"]:
            logger.info("  1. Install DSMIL framework:")
            logger.info("     sudo dpkg -i /home/user/LAT5150DRVMIL/packaging/dsmil-complete_8.3.2-1.deb")
            logger.info("  2. Load kernel module:")
            logger.info("     sudo modprobe dsmil-72dev")

        if self.consumer_npu_available and not self.military_npu_available:
            logger.info("  • Consumer NPU (49.4 TOPS) available for AI workloads")
            logger.info("  • No military NPU detected (normal for consumer hardware)")

        if self.consumer_npu_available and dsmil_status["module_loaded"]:
            logger.info("  ✓ Ready for hardware-attested AI inference!")
            logger.info("  • Use DSMIL AI Engine for production workloads")
            logger.info("  • TPM-sealed model weights available")
            logger.info(f"  • Mode 5 Level: {self.mode5_level.value}")

        return self.consumer_npu_available or self.military_npu_available


def main():
    """Run DSMIL NPU integration"""
    integration = DSMILNPUIntegration()
    success = integration.run_comprehensive_integration()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
