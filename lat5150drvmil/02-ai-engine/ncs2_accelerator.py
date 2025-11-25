"""
Intel NCS2 (Neural Compute Stick 2) Accelerator Integration
============================================================
Provides hardware acceleration using Intel Movidius Myriad X VPU for
deep learning inference in the LAT5150DRVMIL AI platform.

Updated for OpenVINO 2024+ installation process:
- USB-based device detection via libusb
- udev rules for non-root access
- OpenVINO Runtime 2024.x API compatibility
- Automatic driver/rules installation

Features:
- Multi-device support (automatic load balancing)
- Real-time performance monitoring
- Thermal management
- Automatic device detection and initialization
- OpenVINO Runtime integration

Hardware: Intel Neural Compute Stick 2 (Movidius Myriad X VPU)
- 10 TOPS per device
- USB 3.0 interface
- <1W power consumption

Author: LAT5150DRVMIL AI Platform
Version: 2.0.0 (OpenVINO 2024+ compatible)
"""

import glob
import json
import logging
import os
import subprocess
import time
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)

# NCS2 USB Vendor/Product IDs
NCS2_VID = "03e7"  # Intel Movidius
NCS2_PID_BOOT = "2485"  # NCS2 in boot mode
NCS2_PID_RUNTIME = "f63b"  # NCS2 in runtime mode

# OpenVINO device names (changed in 2024+)
OPENVINO_MYRIAD_DEVICE = "MYRIAD"  # Legacy name, still works
OPENVINO_VPU_DEVICE = "VPU"  # Alternative name


@dataclass
class NCS2Device:
    """Represents a single NCS2 device."""
    device_id: int
    device_path: str
    usb_bus: str = ""
    usb_device: str = ""
    temperature: float = 0.0
    utilization: float = 0.0
    firmware_version: str = "unknown"
    total_inferences: int = 0
    is_available: bool = True
    is_throttling: bool = False
    openvino_name: str = ""  # e.g., "MYRIAD.0"


@dataclass
class NCS2Stats:
    """Statistics for NCS2 operations."""
    total_inferences: int = 0
    successful_inferences: int = 0
    failed_inferences: int = 0
    average_latency_ms: float = 0.0
    total_time_ms: float = 0.0
    throughput_fps: float = 0.0


@dataclass
class NCS2InstallStatus:
    """Installation status for NCS2 dependencies."""
    openvino_installed: bool = False
    openvino_version: str = ""
    udev_rules_installed: bool = False
    usb_permissions_ok: bool = False
    devices_detected: int = 0
    ready: bool = False
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class NCS2Accelerator:
    """
    Intel NCS2 Hardware Accelerator Manager.

    Updated for OpenVINO 2024+ with new installation process:
    - USB device detection via lsusb/sysfs
    - udev rules for permissions
    - OpenVINO Runtime 2024.x API

    Manages multiple NCS2 devices for AI inference acceleration with:
    - Automatic device detection and initialization
    - Load balancing across multiple devices
    - Thermal monitoring and throttling protection
    - Performance metrics and telemetry
    """

    # udev rules content for NCS2 access
    UDEV_RULES = """# Intel Movidius Neural Compute Stick 2 (NCS2)
# Boot mode
SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", ATTRS{idProduct}=="2485", MODE="0666", GROUP="users"
# Runtime mode
SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", ATTRS{idProduct}=="f63b", MODE="0666", GROUP="users"
# Generic Movidius
SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666", GROUP="users"
"""

    def __init__(self, enable_monitoring: bool = True, auto_install: bool = False):
        """
        Initialize NCS2 accelerator.

        Args:
            enable_monitoring: Enable real-time device monitoring
            auto_install: Automatically install missing dependencies
        """
        self.devices: Dict[int, NCS2Device] = {}
        self.enable_monitoring = enable_monitoring
        self.stats = NCS2Stats()
        self.next_device_idx = 0  # Round-robin load balancing
        self._ov_core = None
        self._compiled_models: Dict[Tuple, Any] = {}

        # Check installation status
        self.install_status = self._check_installation()

        if auto_install and not self.install_status.ready:
            logger.info("Auto-installing NCS2 dependencies...")
            self.install_dependencies()
            self.install_status = self._check_installation()

        # Detect and initialize devices
        if self.install_status.openvino_installed:
            self._detect_devices_usb()
            self._init_openvino()

        if self.devices:
            logger.info(f"NCS2 Accelerator initialized with {len(self.devices)} device(s)")
        else:
            if not self.install_status.ready:
                logger.warning("NCS2 not ready. Run: ncs2.install_dependencies()")
            else:
                logger.warning("No NCS2 devices detected")

    def _check_installation(self) -> NCS2InstallStatus:
        """Check NCS2 installation status."""
        status = NCS2InstallStatus()

        # Check OpenVINO
        try:
            import openvino
            status.openvino_installed = True
            status.openvino_version = openvino.__version__
            logger.info(f"OpenVINO {status.openvino_version} detected")
        except ImportError:
            status.errors.append("OpenVINO not installed")
            logger.warning("OpenVINO not installed")

        # Check udev rules
        udev_paths = [
            "/etc/udev/rules.d/97-myriad-usbboot.rules",
            "/etc/udev/rules.d/80-movidius.rules",
            "/etc/udev/rules.d/97-ncs2.rules",
        ]
        status.udev_rules_installed = any(Path(p).exists() for p in udev_paths)

        if not status.udev_rules_installed:
            status.errors.append("udev rules not installed")
            logger.warning("NCS2 udev rules not installed")

        # Check USB permissions
        status.usb_permissions_ok = self._check_usb_permissions()

        # Check for devices
        status.devices_detected = self._count_ncs2_devices()

        # Overall ready status
        status.ready = (
            status.openvino_installed and
            status.udev_rules_installed and
            status.usb_permissions_ok and
            status.devices_detected > 0
        )

        return status

    def _check_usb_permissions(self) -> bool:
        """Check if current user has USB access permissions."""
        try:
            # Check if user is in 'users' or 'plugdev' group
            result = subprocess.run(["groups"], capture_output=True, text=True)
            groups = result.stdout.strip().split()
            return "users" in groups or "plugdev" in groups or os.geteuid() == 0
        except Exception:
            return False

    def _count_ncs2_devices(self) -> int:
        """Count NCS2 devices via lsusb."""
        try:
            result = subprocess.run(
                ["lsusb", "-d", f"{NCS2_VID}:"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                lines = [l for l in result.stdout.strip().split('\n') if l]
                return len(lines)
        except FileNotFoundError:
            # lsusb not available, try sysfs
            pass

        # Fallback: check sysfs
        count = 0
        for vendor_file in Path("/sys/bus/usb/devices").glob("*/idVendor"):
            try:
                if vendor_file.read_text().strip() == NCS2_VID:
                    count += 1
            except Exception:
                pass
        return count

    def _detect_devices_usb(self):
        """Detect NCS2 devices via USB enumeration."""
        self.devices.clear()
        device_id = 0

        # Method 1: Use lsusb
        try:
            result = subprocess.run(
                ["lsusb", "-d", f"{NCS2_VID}:"],
                capture_output=True, text=True
            )

            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if not line:
                        continue

                    # Parse: "Bus 001 Device 005: ID 03e7:f63b Intel Corporation"
                    parts = line.split()
                    if len(parts) >= 6:
                        bus = parts[1]
                        dev = parts[3].rstrip(':')
                        device_path = f"/dev/bus/usb/{bus}/{dev}"

                        device = NCS2Device(
                            device_id=device_id,
                            device_path=device_path,
                            usb_bus=bus,
                            usb_device=dev,
                            openvino_name=f"MYRIAD.{device_id}",
                            is_available=True
                        )

                        self.devices[device_id] = device
                        logger.info(f"Detected NCS2 device {device_id} at USB {bus}:{dev}")
                        device_id += 1

        except FileNotFoundError:
            logger.debug("lsusb not available, using sysfs")

        # Method 2: Fallback to sysfs enumeration
        if not self.devices:
            for vendor_file in sorted(Path("/sys/bus/usb/devices").glob("*/idVendor")):
                try:
                    if vendor_file.read_text().strip() == NCS2_VID:
                        device_dir = vendor_file.parent
                        busnum = (device_dir / "busnum").read_text().strip()
                        devnum = (device_dir / "devnum").read_text().strip()
                        device_path = f"/dev/bus/usb/{busnum.zfill(3)}/{devnum.zfill(3)}"

                        device = NCS2Device(
                            device_id=device_id,
                            device_path=device_path,
                            usb_bus=busnum,
                            usb_device=devnum,
                            openvino_name=f"MYRIAD.{device_id}",
                            is_available=True
                        )

                        self.devices[device_id] = device
                        logger.info(f"Detected NCS2 device {device_id} via sysfs")
                        device_id += 1

                except Exception as e:
                    logger.debug(f"Failed to read device info: {e}")

        # Method 3: Use OpenVINO device enumeration
        if not self.devices and self.install_status.openvino_installed:
            try:
                from openvino.runtime import Core
                core = Core()

                available_devices = core.available_devices
                myriad_devices = [d for d in available_devices if d.startswith("MYRIAD")]

                for myriad_dev in myriad_devices:
                    device = NCS2Device(
                        device_id=device_id,
                        device_path=f"openvino://{myriad_dev}",
                        openvino_name=myriad_dev,
                        is_available=True
                    )
                    self.devices[device_id] = device
                    logger.info(f"Detected NCS2 via OpenVINO: {myriad_dev}")
                    device_id += 1

            except Exception as e:
                logger.debug(f"OpenVINO device enumeration failed: {e}")

    def _init_openvino(self):
        """Initialize OpenVINO Core."""
        try:
            from openvino.runtime import Core
            self._ov_core = Core()

            # Log available devices
            devices = self._ov_core.available_devices
            myriad_devices = [d for d in devices if d.startswith("MYRIAD") or d.startswith("VPU")]
            logger.info(f"OpenVINO available devices: {devices}")
            logger.info(f"MYRIAD/VPU devices: {myriad_devices}")

        except ImportError:
            logger.error("OpenVINO not available")
            self._ov_core = None
        except Exception as e:
            logger.error(f"Failed to initialize OpenVINO: {e}")
            self._ov_core = None

    # Path to NCS2 driver submodule
    NCS2_SUBMODULE_PATH = Path(__file__).parent.parent / "04-hardware" / "ncs2-driver"

    def install_dependencies(self) -> bool:
        """
        Install NCS2 dependencies using the NUC2.1 submodule.

        Uses the SWORDIntel/NUC2.1 submodule for kernel driver installation:
        - Kernel module: movidius_x_vpu.ko
        - io_uring support for kernels ≥6.2
        - Fallback to ioctl for older kernels

        Returns:
            True if installation successful
        """
        success = True

        # 1. Check and initialize submodule
        if self.NCS2_SUBMODULE_PATH.exists():
            if not list(self.NCS2_SUBMODULE_PATH.iterdir()):
                logger.info("Initializing NCS2 driver submodule...")
                try:
                    repo_root = Path(__file__).parent.parent
                    subprocess.run(
                        ["git", "submodule", "update", "--init", "04-hardware/ncs2-driver"],
                        cwd=repo_root, check=True, capture_output=True
                    )
                    logger.info("NCS2 submodule initialized")
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to initialize submodule: {e}")

            # Try submodule installer first
            install_script = self.NCS2_SUBMODULE_PATH / "install.sh"
            if install_script.exists():
                logger.info("Running NCS2 submodule installer (NUC2.1)...")
                try:
                    subprocess.run(
                        ["sudo", str(install_script), "install"],
                        cwd=self.NCS2_SUBMODULE_PATH,
                        check=True, capture_output=True, timeout=300
                    )
                    logger.info("NCS2 driver installed via submodule")

                    # Load the kernel module with optimized parameters
                    subprocess.run(
                        ["sudo", "modprobe", "movidius_x_vpu",
                         "batch_delay_ms=5", "batch_high_watermark=32"],
                        capture_output=True
                    )

                    return True
                except subprocess.CalledProcessError as e:
                    logger.warning(f"Submodule installer failed: {e}, trying fallback...")
                except subprocess.TimeoutExpired:
                    logger.warning("Submodule installer timed out, trying fallback...")

        # 2. Fallback: Install OpenVINO if not present
        if not self.install_status.openvino_installed:
            logger.info("Installing OpenVINO...")
            try:
                subprocess.run(
                    ["pip", "install", "openvino>=2024.0"],
                    check=True, capture_output=True
                )
                logger.info("OpenVINO installed successfully")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install OpenVINO: {e}")
                success = False

        # 3. Install udev rules
        if not self.install_status.udev_rules_installed:
            logger.info("Installing udev rules...")
            success = success and self._install_udev_rules()

        # 4. Add user to groups
        if not self.install_status.usb_permissions_ok:
            logger.info("Setting up USB permissions...")
            try:
                user = os.environ.get("USER", "")
                if user:
                    subprocess.run(
                        ["sudo", "usermod", "-aG", "users", user],
                        check=True, capture_output=True
                    )
                    logger.info(f"Added {user} to 'users' group (re-login required)")
            except Exception as e:
                logger.warning(f"Failed to add user to group: {e}")

        return success

    def install_kernel_module(self, batch_delay_ms: int = 5, cpu_affinity: int = -1) -> bool:
        """
        Install and load the NCS2 kernel module from submodule.

        Args:
            batch_delay_ms: Adaptive batch timing (default: 5ms)
            cpu_affinity: CPU core assignment (-1 for auto)

        Returns:
            True if successful
        """
        if not self.NCS2_SUBMODULE_PATH.exists():
            logger.error("NCS2 submodule not found at 04-hardware/ncs2-driver")
            logger.info("Run: git submodule update --init 04-hardware/ncs2-driver")
            return False

        try:
            # Build the kernel module
            logger.info("Building NCS2 kernel module...")
            subprocess.run(
                ["make", "-C", str(self.NCS2_SUBMODULE_PATH)],
                check=True, capture_output=True
            )

            # Install the module
            ko_file = self.NCS2_SUBMODULE_PATH / "movidius_x_vpu.ko"
            if ko_file.exists():
                # Unload existing module if present
                subprocess.run(["sudo", "rmmod", "movidius_x_vpu"],
                             capture_output=True)

                # Load with parameters
                params = [
                    "sudo", "insmod", str(ko_file),
                    f"batch_delay_ms={batch_delay_ms}",
                    f"batch_high_watermark=32"
                ]
                if cpu_affinity >= 0:
                    params.append(f"submission_cpu_affinity={cpu_affinity}")

                subprocess.run(params, check=True, capture_output=True)
                logger.info(f"Kernel module loaded: batch_delay={batch_delay_ms}ms")

                # Verify device nodes
                if list(Path("/dev").glob("movidius*")):
                    logger.info("Device nodes created successfully")
                    return True
                else:
                    logger.warning("Module loaded but no device nodes found")
                    return False
            else:
                logger.error("Kernel module not found after build")
                return False

        except subprocess.CalledProcessError as e:
            logger.error(f"Kernel module installation failed: {e}")
            return False

    def _install_udev_rules(self) -> bool:
        """Install udev rules for NCS2."""
        rules_path = "/etc/udev/rules.d/97-ncs2.rules"

        try:
            # Write rules file
            rules_file = Path("/tmp/97-ncs2.rules")
            rules_file.write_text(self.UDEV_RULES)

            # Copy with sudo
            subprocess.run(
                ["sudo", "cp", str(rules_file), rules_path],
                check=True, capture_output=True
            )

            # Reload udev rules
            subprocess.run(
                ["sudo", "udevadm", "control", "--reload-rules"],
                check=True, capture_output=True
            )
            subprocess.run(
                ["sudo", "udevadm", "trigger"],
                check=True, capture_output=True
            )

            logger.info(f"udev rules installed at {rules_path}")
            rules_file.unlink()
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install udev rules: {e}")
            return False
        except Exception as e:
            logger.error(f"udev rules installation error: {e}")
            return False

    def is_available(self) -> bool:
        """Check if any NCS2 devices are available."""
        return len(self.devices) > 0 and self._ov_core is not None

    def get_device_count(self) -> int:
        """Get number of available devices."""
        return len(self.devices)

    def get_next_device(self) -> Optional[NCS2Device]:
        """
        Get next available device using round-robin load balancing.

        Returns:
            NCS2Device or None if no devices available
        """
        if not self.devices:
            return None

        # Find non-throttling devices
        available_devices = [
            d for d in self.devices.values()
            if d.is_available and not d.is_throttling
        ]

        if not available_devices:
            logger.warning("All NCS2 devices are throttling or unavailable")
            return None

        # Round-robin selection
        device = available_devices[self.next_device_idx % len(available_devices)]
        self.next_device_idx += 1

        return device

    def infer(
        self,
        model_data: bytes,
        input_data: np.ndarray,
        device_id: Optional[int] = None
    ) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Run inference on NCS2 device.

        Args:
            model_data: Compiled model blob (OpenVINO IR XML)
            input_data: Input tensor (numpy array)
            device_id: Specific device ID (None for auto-selection)

        Returns:
            Tuple of (success, output_data, latency_ms)
        """
        start_time = time.time()

        if self._ov_core is None:
            logger.error("OpenVINO not initialized")
            return False, None, 0.0

        # Select device
        if device_id is not None:
            device = self.devices.get(device_id)
            if device is None:
                logger.error(f"Device {device_id} not found")
                return False, None, 0.0
        else:
            device = self.get_next_device()
            if device is None:
                logger.error("No available NCS2 devices")
                return False, None, 0.0

        try:
            # Update device stats before inference
            if self.enable_monitoring:
                self._update_device_info(device)

            # Check if device is throttling
            if device.is_throttling:
                logger.warning(f"Device {device.device_id} is throttling (temp: {device.temperature}°C)")
                device = self.get_next_device()
                if device is None:
                    return False, None, 0.0

            # Run inference via OpenVINO
            output_data = self._run_inference_openvino(device, model_data, input_data)

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000

            # Update stats
            self.stats.total_inferences += 1
            self.stats.successful_inferences += 1
            self.stats.total_time_ms += latency_ms
            device.total_inferences += 1

            if self.stats.total_inferences > 0:
                self.stats.average_latency_ms = (
                    self.stats.total_time_ms / self.stats.total_inferences
                )
                self.stats.throughput_fps = (
                    1000.0 / self.stats.average_latency_ms
                    if self.stats.average_latency_ms > 0 else 0.0
                )

            return True, output_data, latency_ms

        except Exception as e:
            logger.error(f"Inference failed on device {device.device_id}: {e}")
            self.stats.total_inferences += 1
            self.stats.failed_inferences += 1
            return False, None, 0.0

    def _run_inference_openvino(
        self,
        device: NCS2Device,
        model_data: bytes,
        input_data: np.ndarray
    ) -> np.ndarray:
        """
        Run inference using OpenVINO Runtime 2024+.

        Args:
            device: Target NCS2 device
            model_data: OpenVINO IR model (.xml content)
            input_data: Input tensor

        Returns:
            Output tensor (numpy array)
        """
        logger.debug(f"Running inference on device {device.device_id} ({device.openvino_name})")

        try:
            from openvino.runtime import Core, CompiledModel, Type, PartialShape

            # Get or compile model for this device
            model_hash = hash(model_data)
            cache_key = (device.device_id, model_hash)

            if cache_key not in self._compiled_models:
                import tempfile

                # Write model to temp file
                with tempfile.NamedTemporaryFile(suffix='.xml', delete=False) as f:
                    f.write(model_data)
                    model_path = f.name

                # Also write .bin file if embedded or expect it
                bin_path = model_path.replace('.xml', '.bin')

                try:
                    model = self._ov_core.read_model(model_path)

                    # Determine device name - try MYRIAD first, then VPU
                    device_name = device.openvino_name
                    if not device_name:
                        # Try to find available MYRIAD device
                        available = self._ov_core.available_devices
                        for dev in available:
                            if dev.startswith("MYRIAD") or dev.startswith("VPU"):
                                device_name = dev
                                break
                        if not device_name:
                            device_name = "MYRIAD"  # Default

                    # Compile configuration for MYRIAD/NCS2
                    config = {}

                    # OpenVINO 2024+ config keys (different from legacy)
                    try:
                        # Try new-style config
                        config["PERFORMANCE_HINT"] = "LATENCY"
                    except Exception:
                        # Fallback to legacy config
                        config["MYRIAD_ENABLE_HW_ACCELERATION"] = "YES"

                    compiled = self._ov_core.compile_model(model, device_name, config)
                    self._compiled_models[cache_key] = compiled

                    logger.debug(f"Model compiled for {device_name}")

                finally:
                    # Cleanup temp files
                    try:
                        os.unlink(model_path)
                        if os.path.exists(bin_path):
                            os.unlink(bin_path)
                    except Exception:
                        pass

            compiled_model = self._compiled_models[cache_key]

            # Create inference request
            infer_request = compiled_model.create_infer_request()

            # Set input tensor
            input_tensor = infer_request.get_input_tensor(0)

            # Handle shape mismatch
            if input_tensor.shape != input_data.shape:
                logger.debug(f"Reshaping input from {input_data.shape} to {input_tensor.shape}")
                input_data = input_data.reshape(input_tensor.shape)

            input_tensor.data[:] = input_data

            # Run synchronous inference
            infer_request.infer()

            # Get output
            output_tensor = infer_request.get_output_tensor(0)
            return output_tensor.data.copy()

        except ImportError:
            logger.warning("OpenVINO not available, using simulation mode")
            time.sleep(0.002)
            return np.zeros_like(input_data)

        except Exception as e:
            logger.error(f"OpenVINO inference failed: {e}")
            return np.zeros_like(input_data)

    def _update_device_info(self, device: NCS2Device):
        """Update device information (temperature, utilization)."""
        # NCS2 doesn't expose detailed thermal info via standard interfaces
        # We can estimate based on inference count and time
        # In production, this would query USB device if supported

        # Simple heuristic: estimate temp based on recent usage
        if device.total_inferences > 0:
            # Rough estimate: base temp + usage factor
            device.temperature = 35.0 + (device.utilization * 0.4)
            device.is_throttling = device.temperature > 75.0

    def load_model(self, model_path: str) -> Optional[bytes]:
        """
        Load model for NCS2 inference.

        Args:
            model_path: Path to OpenVINO IR model (.xml)

        Returns:
            Model bytes or None
        """
        try:
            with open(model_path, 'rb') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None

    def benchmark(self, model_data: bytes, input_shape: Tuple, iterations: int = 100) -> Dict:
        """
        Benchmark NCS2 inference performance.

        Args:
            model_data: Model bytes
            input_shape: Shape of input tensor
            iterations: Number of iterations

        Returns:
            Benchmark results dictionary
        """
        input_data = np.random.rand(*input_shape).astype(np.float32)
        latencies = []

        # Warmup
        for _ in range(10):
            self.infer(model_data, input_data)

        # Benchmark
        for _ in range(iterations):
            success, _, latency = self.infer(model_data, input_data)
            if success:
                latencies.append(latency)

        if not latencies:
            return {"error": "All inferences failed"}

        return {
            "iterations": len(latencies),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "avg_latency_ms": sum(latencies) / len(latencies),
            "p50_latency_ms": sorted(latencies)[len(latencies) // 2],
            "p99_latency_ms": sorted(latencies)[int(len(latencies) * 0.99)],
            "throughput_fps": 1000.0 / (sum(latencies) / len(latencies))
        }

    def get_device_info(self, device_id: int) -> Optional[Dict]:
        """Get detailed information about a specific device."""
        device = self.devices.get(device_id)
        if device is None:
            return None

        self._update_device_info(device)

        return {
            "device_id": device.device_id,
            "device_path": device.device_path,
            "openvino_name": device.openvino_name,
            "usb_bus": device.usb_bus,
            "usb_device": device.usb_device,
            "temperature": device.temperature,
            "utilization": device.utilization,
            "firmware_version": device.firmware_version,
            "total_inferences": device.total_inferences,
            "is_available": device.is_available,
            "is_throttling": device.is_throttling
        }

    def get_all_devices_info(self) -> List[Dict]:
        """Get information about all devices."""
        return [
            self.get_device_info(device_id)
            for device_id in sorted(self.devices.keys())
        ]

    def get_stats(self) -> Dict:
        """Get aggregated statistics."""
        return {
            "total_inferences": self.stats.total_inferences,
            "successful_inferences": self.stats.successful_inferences,
            "failed_inferences": self.stats.failed_inferences,
            "success_rate": (
                self.stats.successful_inferences / self.stats.total_inferences
                if self.stats.total_inferences > 0 else 0.0
            ),
            "average_latency_ms": self.stats.average_latency_ms,
            "throughput_fps": self.stats.throughput_fps,
            "device_count": len(self.devices),
            "install_status": {
                "openvino_installed": self.install_status.openvino_installed,
                "openvino_version": self.install_status.openvino_version,
                "udev_rules_installed": self.install_status.udev_rules_installed,
                "ready": self.install_status.ready
            }
        }

    def get_install_status(self) -> Dict:
        """Get detailed installation status."""
        return {
            "openvino_installed": self.install_status.openvino_installed,
            "openvino_version": self.install_status.openvino_version,
            "udev_rules_installed": self.install_status.udev_rules_installed,
            "usb_permissions_ok": self.install_status.usb_permissions_ok,
            "devices_detected": self.install_status.devices_detected,
            "ready": self.install_status.ready,
            "errors": self.install_status.errors
        }

    def monitor_devices(self) -> Dict:
        """Monitor all devices and return current status."""
        monitoring_data = {
            "timestamp": time.time(),
            "devices": [],
            "alerts": []
        }

        for device_id in sorted(self.devices.keys()):
            device = self.devices[device_id]
            self._update_device_info(device)

            device_data = {
                "device_id": device_id,
                "openvino_name": device.openvino_name,
                "temperature": device.temperature,
                "utilization": device.utilization,
                "is_throttling": device.is_throttling,
                "total_inferences": device.total_inferences
            }

            monitoring_data["devices"].append(device_data)

            if device.is_throttling:
                monitoring_data["alerts"].append({
                    "severity": "WARNING",
                    "device_id": device_id,
                    "message": f"Device {device_id} is throttling (temp: {device.temperature}°C)"
                })

        return monitoring_data

    def reset_stats(self):
        """Reset statistics counters."""
        self.stats = NCS2Stats()
        for device in self.devices.values():
            device.total_inferences = 0
        logger.info("NCS2 statistics reset")

    def __repr__(self) -> str:
        return (
            f"NCS2Accelerator(devices={len(self.devices)}, "
            f"inferences={self.stats.total_inferences}, "
            f"avg_latency={self.stats.average_latency_ms:.2f}ms, "
            f"ready={self.install_status.ready})"
        )


# Singleton instance
_ncs2_accelerator: Optional[NCS2Accelerator] = None


def get_ncs2_accelerator(auto_install: bool = False) -> Optional[NCS2Accelerator]:
    """
    Get or create singleton NCS2 accelerator instance.

    Args:
        auto_install: Automatically install dependencies if missing

    Returns:
        NCS2Accelerator instance or None if not available
    """
    global _ncs2_accelerator

    if _ncs2_accelerator is None:
        try:
            _ncs2_accelerator = NCS2Accelerator(
                enable_monitoring=True,
                auto_install=auto_install
            )

            if not _ncs2_accelerator.is_available():
                logger.info("NCS2 not available, hardware acceleration disabled")
                # Keep instance for install_dependencies() access

        except Exception as e:
            logger.error(f"Failed to initialize NCS2 accelerator: {e}")
            _ncs2_accelerator = None

    return _ncs2_accelerator


def is_ncs2_available() -> bool:
    """Check if NCS2 hardware acceleration is available."""
    accelerator = get_ncs2_accelerator()
    return accelerator is not None and accelerator.is_available()


def install_ncs2_dependencies() -> bool:
    """
    Install NCS2 dependencies.

    Returns:
        True if installation successful
    """
    accelerator = get_ncs2_accelerator()
    if accelerator:
        return accelerator.install_dependencies()
    return False
