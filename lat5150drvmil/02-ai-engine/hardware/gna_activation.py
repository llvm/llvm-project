#!/usr/bin/env python3
"""
Intel GNA (Gaussian & Neural-Network Accelerator) Activation Script

Activates and tests the Intel Meteor Lake GNA

Hardware Detected:
- Device: 0000:00:08.0 Intel Meteor Lake-P GNA [8086:7e4c] (rev 20)
- Status: DISABLED (IRQ 255, no driver loaded)
- Memory: 501c2e3000 (64-bit, non-prefetchable) [disabled] [size=4K]

GNA Capabilities:
- Low-power neural network inference
- Audio processing (noise reduction, wake word detection)
- Speech recognition acceleration
- Always-on inference (ultra-low power)

Activation Steps:
1. Enable PCI device
2. Load GNA driver (if available)
3. Verify device activation
4. Run basic tests
"""

import os
import sys
import subprocess
import time
import logging
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GNAActivator:
    """Activate and test Intel GNA"""

    def __init__(self):
        self.pci_address = "0000:00:08.0"
        self.pci_id = "8086:7e4c"
        self.driver_loaded = False
        self.device_enabled = False

    def check_pci_device(self) -> bool:
        """Check if GNA PCI device exists"""
        logger.info("Checking GNA PCI device...")

        try:
            result = subprocess.run(
                ["lspci", "-s", self.pci_address, "-v"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0 and self.pci_id in result.stdout:
                logger.info(f"✓ GNA PCI device found: {self.pci_address}")

                # Check if disabled
                if "[disabled]" in result.stdout:
                    logger.warning("⚠️  Device is DISABLED")
                    return True  # Present but disabled
                else:
                    logger.info("✓ Device is ENABLED")
                    self.device_enabled = True
                    return True
            else:
                logger.error(f"✗ GNA device not found at {self.pci_address}")
                return False

        except Exception as e:
            logger.error(f"✗ Error checking PCI device: {e}")
            return False

    def enable_pci_device(self) -> bool:
        """Enable GNA PCI device"""
        if self.device_enabled:
            logger.info("Device already enabled")
            return True

        logger.info("Attempting to enable GNA PCI device...")

        try:
            # Method 1: Using setpci to enable device
            logger.info("Method 1: Using setpci...")

            # Read current command register
            result = subprocess.run(
                ["sudo", "setpci", "-s", self.pci_address, "COMMAND"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                current_cmd = result.stdout.strip()
                logger.info(f"  Current COMMAND register: {current_cmd}")

                # Set bit 1 (Memory Space Enable) and bit 2 (Bus Master Enable)
                # Typical value: 0x0006 (bits 1 and 2 set)
                new_cmd = "0x0006"

                logger.info(f"  Setting COMMAND register to: {new_cmd}")
                subprocess.run(
                    ["sudo", "setpci", "-s", self.pci_address, f"COMMAND={new_cmd}"],
                    check=True,
                    timeout=5
                )

                logger.info("✓ PCI command register updated")

                # Verify
                time.sleep(1)
                if self.check_pci_device():
                    logger.info("✓ Device enabled successfully")
                    return True

        except subprocess.CalledProcessError as e:
            logger.warning(f"⚠️  Method 1 failed: {e}")

        except Exception as e:
            logger.warning(f"⚠️  Method 1 failed: {e}")

        # Method 2: Using PCI rescan
        logger.info("Method 2: PCI rescan...")
        try:
            subprocess.run(
                ["sudo", "sh", "-c", "echo 1 > /sys/bus/pci/rescan"],
                check=True,
                timeout=5
            )

            time.sleep(2)

            if self.check_pci_device():
                logger.info("✓ Device enabled via PCI rescan")
                return True

        except Exception as e:
            logger.warning(f"⚠️  Method 2 failed: {e}")

        logger.error("✗ Failed to enable device")
        return False

    def check_available_drivers(self) -> List[str]:
        """Check for available GNA drivers"""
        logger.info("Checking for GNA drivers...")

        drivers = []

        # Check loaded modules
        try:
            result = subprocess.run(
                ["lsmod"],
                capture_output=True,
                text=True,
                timeout=5
            )

            # Known GNA driver names
            gna_drivers = ["intel_gna", "gna", "snd_hda_intel"]

            for driver in gna_drivers:
                if driver in result.stdout:
                    logger.info(f"✓ Driver loaded: {driver}")
                    drivers.append(driver)
                    self.driver_loaded = True

        except:
            pass

        # Check available modules
        try:
            result = subprocess.run(
                ["modprobe", "-l"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if "gna" in result.stdout.lower():
                logger.info("✓ GNA driver modules available")

        except:
            pass

        if not drivers:
            logger.warning("⚠️  No GNA drivers currently loaded")
            logger.info("   GNA may be integrated with audio subsystem (snd_hda_intel)")

        return drivers

    def attempt_driver_load(self) -> bool:
        """Attempt to load GNA driver"""
        logger.info("Attempting to load GNA driver...")

        # GNA is often part of the audio subsystem on Meteor Lake
        drivers_to_try = [
            "intel_gna",
            "gna",
        ]

        for driver in drivers_to_try:
            try:
                logger.info(f"Trying: modprobe {driver}")
                result = subprocess.run(
                    ["sudo", "modprobe", driver],
                    capture_output=True,
                    text=True,
                    timeout=5
                )

                if result.returncode == 0:
                    logger.info(f"✓ Loaded driver: {driver}")
                    self.driver_loaded = True
                    return True
                else:
                    logger.warning(f"⚠️  Failed to load {driver}: {result.stderr}")

            except Exception as e:
                logger.warning(f"⚠️  Error loading {driver}: {e}")

        logger.warning("⚠️  Could not load standalone GNA driver")
        logger.info("   Note: GNA may be integrated with NPU or audio subsystem")
        return False

    def check_openvino_gna(self) -> bool:
        """Check if GNA is accessible via OpenVINO"""
        logger.info("Checking OpenVINO GNA support...")

        try:
            from openvino.runtime import Core

            core = Core()
            devices = core.available_devices

            logger.info(f"Available devices: {devices}")

            if "GNA" in devices:
                logger.info("✓ GNA detected by OpenVINO")
                return True
            else:
                logger.warning("⚠️  GNA not detected by OpenVINO")
                logger.info("   This is expected - GNA support may require specific OpenVINO build")
                return False

        except ImportError:
            logger.warning("⚠️  OpenVINO not installed")
            return False
        except Exception as e:
            logger.error(f"✗ Error checking OpenVINO: {e}")
            return False

    def get_device_info(self) -> Dict:
        """Get detailed GNA device information"""
        logger.info("\nGNA Device Information:")
        logger.info("=" * 80)

        try:
            result = subprocess.run(
                ["lspci", "-s", self.pci_address, "-vvv"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                print(result.stdout)

                info = {
                    "pci_address": self.pci_address,
                    "pci_id": self.pci_id,
                    "enabled": self.device_enabled,
                    "driver_loaded": self.driver_loaded,
                }

                return info

        except:
            pass

        return {}

    def run_activation(self) -> bool:
        """Run complete GNA activation process"""
        logger.info("\n" + "=" * 80)
        logger.info("  Intel GNA Activation")
        logger.info("=" * 80)
        logger.info("Hardware: Dell Latitude 5450 MIL-SPEC")
        logger.info("GNA: Intel Meteor Lake-P GNA [8086:7e4c]")
        logger.info("=" * 80)

        # Step 1: Check PCI device
        logger.info("\n[Step 1/5] Checking PCI Device")
        pci_ok = self.check_pci_device()

        if not pci_ok:
            logger.error("✗ GNA device not found. Aborting.")
            return False

        # Step 2: Enable device (if disabled)
        logger.info("\n[Step 2/5] Enabling Device")
        if not self.device_enabled:
            enable_ok = self.enable_pci_device()
        else:
            enable_ok = True
            logger.info("Device already enabled")

        # Step 3: Check drivers
        logger.info("\n[Step 3/5] Checking Drivers")
        drivers = self.check_available_drivers()

        # Step 4: Load driver
        logger.info("\n[Step 4/5] Loading Driver")
        if not self.driver_loaded:
            driver_ok = self.attempt_driver_load()
        else:
            driver_ok = True
            logger.info("Driver already loaded")

        # Step 5: OpenVINO check
        logger.info("\n[Step 5/5] OpenVINO GNA Check")
        openvino_ok = self.check_openvino_gna()

        # Get device info
        logger.info("\n[Device Info]")
        info = self.get_device_info()

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("  Activation Summary")
        logger.info("=" * 80)
        logger.info(f"PCI Device:     {'✓ FOUND' if pci_ok else '✗ NOT FOUND'}")
        logger.info(f"Device Enabled: {'✓ YES' if self.device_enabled else '✗ NO'}")
        logger.info(f"Driver Loaded:  {'✓ YES' if self.driver_loaded else '✗ NO'}")
        logger.info(f"OpenVINO GNA:   {'✓ YES' if openvino_ok else '✗ NO'}")
        logger.info("=" * 80)

        if self.device_enabled:
            logger.info("\n✓ GNA device is ENABLED")
        else:
            logger.warning("\n⚠️  GNA device is still DISABLED")
            logger.info("   This may require:")
            logger.info("   - BIOS setting changes")
            logger.info("   - Kernel boot parameters")
            logger.info("   - Specific firmware/driver version")

        logger.info("\nNote: GNA on Meteor Lake may be:")
        logger.info("  1. Integrated with NPU (unified accelerator)")
        logger.info("  2. Part of audio subsystem (for audio ML)")
        logger.info("  3. Require proprietary driver for activation")

        return pci_ok and self.device_enabled


def main():
    """Run GNA activation"""
    activator = GNAActivator()
    success = activator.run_activation()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
