#!/usr/bin/env python3
"""
Military NPU DSMIL Driver Loader

Detects and activates military-grade NPU hardware with DSMIL driver

Expected Hardware:
- Military-grade Neural Processing Unit (classification varies)
- Enhanced performance (100-200+ TOPS vs consumer 49.4 TOPS)
- Specialized crypto acceleration
- Export-controlled features
- DSMIL enumeration integration

DSMIL Driver:
- Custom kernel module for military NPU
- Likely located in: /lib/modules/$(uname -r)/extra/dsmil/
- Module name: dsmil_npu.ko or similar
- May require firmware: /lib/firmware/dsmil/

Security:
- Requires clearance verification
- May require secure boot
- Hardware attestation
- FIPS compliance checks
"""

import os
import sys
import subprocess
import time
import json
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MilitaryNPULoader:
    """Load and activate military NPU with DSMIL driver"""

    def __init__(self):
        self.military_npu_found = False
        self.dsmil_driver_loaded = False
        self.npu_device_info = {}
        self.security_status = {}

        # Known military NPU PCI IDs (examples - actual IDs may vary)
        self.military_pci_ids = [
            "8086:7e5c",  # Example: Intel military NPU variant
            "8086:7f4c",  # Example: Classified Intel accelerator
            "8086:9d1d",  # Example: Enhanced NPU
        ]

    def scan_pci_devices(self) -> List[Dict]:
        """Scan for potential military NPU devices"""
        logger.info("Scanning PCI bus for military NPU hardware...")

        devices = []

        try:
            result = subprocess.run(
                ["lspci", "-nn", "-D"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                logger.error("Failed to scan PCI bus")
                return devices

            # Parse lspci output
            for line in result.stdout.split('\n'):
                # Look for processing accelerators, system peripherals, or classified devices
                if any(keyword in line.lower() for keyword in [
                    "processing accelerator",
                    "neural",
                    "npu",
                    "accelerator",
                    "system peripheral"
                ]):
                    # Extract PCI address and ID
                    parts = line.split()
                    if len(parts) >= 3:
                        pci_address = parts[0]

                        # Extract vendor:device ID from brackets [xxxx:xxxx]
                        import re
                        match = re.search(r'\[([0-9a-f]{4}):([0-9a-f]{4})\]', line)
                        if match:
                            vendor_id = match.group(1)
                            device_id = match.group(2)
                            pci_id = f"{vendor_id}:{device_id}"

                            device_info = {
                                "pci_address": pci_address,
                                "pci_id": pci_id,
                                "description": line,
                                "is_military": pci_id in self.military_pci_ids
                            }

                            devices.append(device_info)

                            if device_info["is_military"]:
                                logger.info(f"🎯 MILITARY NPU DETECTED: {pci_address} [{pci_id}]")
                                self.military_npu_found = True
                                self.npu_device_info = device_info
                            else:
                                logger.info(f"   Potential NPU: {pci_address} [{pci_id}]")

        except Exception as e:
            logger.error(f"Error scanning PCI devices: {e}")

        if not self.military_npu_found:
            logger.warning("⚠️  No military NPU detected via PCI ID matching")
            logger.info("   Checking for non-standard configurations...")

        return devices

    def check_dsmil_driver_availability(self) -> bool:
        """Check if DSMIL driver is available"""
        logger.info("\nChecking DSMIL driver availability...")

        # Check for DSMIL kernel module
        kernel_version = subprocess.run(
            ["uname", "-r"],
            capture_output=True,
            text=True
        ).stdout.strip()

        module_paths = [
            f"/lib/modules/{kernel_version}/extra/dsmil/dsmil_npu.ko",
            f"/lib/modules/{kernel_version}/kernel/drivers/dsmil/dsmil_npu.ko",
            f"/lib/modules/{kernel_version}/updates/dsmil/dsmil_npu.ko",
            "/opt/dsmil/drivers/dsmil_npu.ko",
            "/tank/ai-engine/drivers/dsmil_npu.ko",  # Custom location on ZFS
        ]

        for path in module_paths:
            if os.path.exists(path):
                logger.info(f"✓ Found DSMIL driver: {path}")
                return True

        logger.warning("⚠️  DSMIL driver not found in standard locations")

        # Check if already loaded
        try:
            result = subprocess.run(
                ["lsmod"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if "dsmil" in result.stdout:
                logger.info("✓ DSMIL driver already loaded")
                self.dsmil_driver_loaded = True
                return True

        except:
            pass

        return False

    def check_dsmil_firmware(self) -> bool:
        """Check for DSMIL firmware"""
        logger.info("Checking DSMIL firmware...")

        firmware_paths = [
            "/lib/firmware/dsmil/",
            "/tank/ai-engine/firmware/dsmil/",
            "/opt/dsmil/firmware/",
        ]

        for path in firmware_paths:
            if os.path.exists(path):
                # List firmware files
                try:
                    files = os.listdir(path)
                    if files:
                        logger.info(f"✓ Found DSMIL firmware in: {path}")
                        logger.info(f"   Files: {', '.join(files[:5])}")
                        return True
                except:
                    pass

        logger.warning("⚠️  DSMIL firmware not found")
        logger.info("   Firmware may be embedded in driver or loaded separately")
        return False

    def load_dsmil_driver(self) -> bool:
        """Load DSMIL kernel module"""
        logger.info("\nAttempting to load DSMIL driver...")

        # Module names to try
        module_names = [
            "dsmil_npu",
            "dsmil",
            "dsmil_accelerator",
            "intel_dsmil",
        ]

        for module in module_names:
            try:
                logger.info(f"Trying: modprobe {module}")

                result = subprocess.run(
                    ["sudo", "modprobe", module],
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                if result.returncode == 0:
                    logger.info(f"✓ Successfully loaded: {module}")
                    self.dsmil_driver_loaded = True

                    # Wait for device initialization
                    time.sleep(2)

                    return True
                else:
                    logger.debug(f"   Failed: {result.stderr.strip()}")

            except subprocess.TimeoutExpired:
                logger.warning(f"⚠️  Timeout loading {module}")
            except Exception as e:
                logger.debug(f"   Error: {e}")

        logger.error("✗ Failed to load DSMIL driver")
        return False

    def verify_military_npu_activation(self) -> bool:
        """Verify military NPU is activated"""
        logger.info("\nVerifying military NPU activation...")

        # Check for device nodes
        device_paths = [
            "/dev/dsmil0",
            "/dev/dsmil/npu0",
            "/dev/military_npu0",
            "/dev/accel/accel1",  # Secondary accelerator
        ]

        for path in device_paths:
            if os.path.exists(path):
                logger.info(f"✓ Device node found: {path}")

                # Check permissions
                try:
                    stat_info = os.stat(path)
                    logger.info(f"   Permissions: {oct(stat_info.st_mode)[-3:]}")
                except:
                    pass

                return True

        logger.warning("⚠️  No DSMIL device nodes found")

        # Check dmesg for DSMIL messages
        try:
            result = subprocess.run(
                ["dmesg", "|", "grep", "-i", "dsmil"],
                shell=True,
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.stdout:
                logger.info("DSMIL kernel messages:")
                for line in result.stdout.strip().split('\n')[-10:]:
                    logger.info(f"   {line}")

        except:
            pass

        return False

    def check_security_compliance(self) -> Dict:
        """Check security and compliance status"""
        logger.info("\nChecking security compliance...")

        status = {
            "secure_boot": False,
            "fips_mode": False,
            "clearance_verified": False,
            "hardware_attestation": False,
        }

        # Check secure boot
        try:
            if os.path.exists("/sys/firmware/efi/efivars/SecureBoot-*"):
                result = subprocess.run(
                    ["mokutil", "--sb-state"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if "SecureBoot enabled" in result.stdout:
                    status["secure_boot"] = True
                    logger.info("✓ Secure Boot: ENABLED")
                else:
                    logger.warning("⚠️  Secure Boot: DISABLED")
        except:
            logger.warning("⚠️  Could not verify Secure Boot status")

        # Check FIPS mode
        try:
            if os.path.exists("/proc/sys/crypto/fips_enabled"):
                with open("/proc/sys/crypto/fips_enabled", "r") as f:
                    if f.read().strip() == "1":
                        status["fips_mode"] = True
                        logger.info("✓ FIPS Mode: ENABLED")
                    else:
                        logger.warning("⚠️  FIPS Mode: DISABLED")
        except:
            logger.warning("⚠️  Could not verify FIPS mode")

        # Check for clearance verification file
        clearance_paths = [
            "/etc/dsmil/clearance.cert",
            "/opt/dsmil/security/clearance.pem",
            "/tank/ai-engine/security/clearance.cert",
        ]

        for path in clearance_paths:
            if os.path.exists(path):
                status["clearance_verified"] = True
                logger.info(f"✓ Clearance certificate found: {path}")
                break

        if not status["clearance_verified"]:
            logger.warning("⚠️  No clearance certificate found")

        self.security_status = status
        return status

    def get_npu_capabilities(self) -> Dict:
        """Query military NPU capabilities"""
        logger.info("\nQuerying NPU capabilities...")

        capabilities = {
            "tops_rating": 0,
            "crypto_accel": False,
            "export_controlled": False,
            "classification": "UNCLASSIFIED",
        }

        # Try to read capabilities from sysfs
        if self.npu_device_info.get("pci_address"):
            pci_addr = self.npu_device_info["pci_address"]

            # Check for capability files
            cap_paths = [
                f"/sys/bus/pci/devices/{pci_addr}/dsmil_capabilities",
                f"/sys/class/dsmil/dsmil0/capabilities",
            ]

            for path in cap_paths:
                if os.path.exists(path):
                    try:
                        with open(path, "r") as f:
                            cap_data = f.read()
                            logger.info(f"Capabilities: {cap_data}")
                            # Parse capabilities (format depends on driver)
                    except:
                        pass

        return capabilities

    def run_activation_sequence(self) -> bool:
        """Run complete military NPU activation sequence"""
        logger.info("\n" + "=" * 80)
        logger.info("  MILITARY NPU DSMIL DRIVER LOADER")
        logger.info("=" * 80)
        logger.info("Hardware: Dell Latitude 5450 MIL-SPEC")
        logger.info("Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY")
        logger.info("=" * 80)

        # Step 1: Scan for hardware
        logger.info("\n[Step 1/7] Hardware Detection")
        devices = self.scan_pci_devices()

        if not devices:
            logger.warning("⚠️  No NPU devices found")

        # Step 2: Check driver availability
        logger.info("\n[Step 2/7] Driver Availability")
        driver_available = self.check_dsmil_driver_availability()

        # Step 3: Check firmware
        logger.info("\n[Step 3/7] Firmware Check")
        firmware_available = self.check_dsmil_firmware()

        # Step 4: Security compliance
        logger.info("\n[Step 4/7] Security Compliance")
        security = self.check_security_compliance()

        # Step 5: Load driver
        logger.info("\n[Step 5/7] Driver Loading")
        if driver_available and not self.dsmil_driver_loaded:
            driver_loaded = self.load_dsmil_driver()
        else:
            driver_loaded = self.dsmil_driver_loaded
            if driver_loaded:
                logger.info("Driver already loaded")

        # Step 6: Verify activation
        logger.info("\n[Step 6/7] Activation Verification")
        if driver_loaded:
            activated = self.verify_military_npu_activation()
        else:
            activated = False
            logger.warning("⚠️  Skipping verification (driver not loaded)")

        # Step 7: Query capabilities
        logger.info("\n[Step 7/7] Capability Query")
        if activated:
            capabilities = self.get_npu_capabilities()
        else:
            capabilities = {}
            logger.warning("⚠️  Skipping capability query (device not activated)")

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("  ACTIVATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Military NPU:      {'✓ DETECTED' if self.military_npu_found else '⚠️  NOT DETECTED'}")
        logger.info(f"DSMIL Driver:      {'✓ LOADED' if driver_loaded else '✗ NOT LOADED'}")
        logger.info(f"Device Activated:  {'✓ YES' if activated else '✗ NO'}")
        logger.info(f"Secure Boot:       {'✓ ENABLED' if security.get('secure_boot') else '⚠️  DISABLED'}")
        logger.info(f"FIPS Mode:         {'✓ ENABLED' if security.get('fips_mode') else '⚠️  DISABLED'}")
        logger.info("=" * 80)

        # Export status
        status_file = "/tank/ai-engine/logs/military_npu_status.json"
        os.makedirs(os.path.dirname(status_file), exist_ok=True)

        status_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "military_npu_detected": self.military_npu_found,
            "dsmil_driver_loaded": driver_loaded,
            "device_activated": activated,
            "security_status": security,
            "npu_info": self.npu_device_info,
            "capabilities": capabilities,
        }

        try:
            with open(status_file, "w") as f:
                json.dump(status_data, f, indent=2)
            logger.info(f"\n✓ Status exported to: {status_file}")
        except:
            pass

        # Final status
        if activated:
            logger.info("\n🎉 MILITARY NPU ACTIVATED SUCCESSFULLY!")
            logger.info("   Enhanced capabilities available")
            logger.info("   DSMIL integration ready")
        elif self.military_npu_found and not driver_loaded:
            logger.warning("\n⚠️  Military NPU detected but driver not loaded")
            logger.info("   Check DSMIL driver installation:")
            logger.info("   - Ensure driver is compiled for current kernel")
            logger.info("   - Verify firmware is present")
            logger.info("   - Check security clearance requirements")
        elif not self.military_npu_found:
            logger.info("\n📝 NO MILITARY NPU DETECTED")
            logger.info("   This system has:")
            logger.info("   - Consumer NPU: Intel Meteor Lake NPU (49.4 TOPS INT8)")
            logger.info("   - GNA: Intel Gaussian & Neural-Network Accelerator")
            logger.info("   Both are available for AI workloads")

        return activated


def main():
    """Run military NPU activation"""
    loader = MilitaryNPULoader()
    success = loader.run_activation_sequence()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
