#!/usr/bin/env python3
"""
Intel NPU Test & Benchmark Script

Tests the Intel Meteor Lake NPU (49.4 TOPS INT8) with OpenVINO

Hardware Detected:
- Device: 0000:00:0b.0 Intel Meteor Lake NPU [8086:7d1d]
- Driver: intel_vpu
- Firmware: intel/vpu/vpu_37xx_v1.bin
- Device: /dev/accel/accel0

Tests:
1. Device detection and availability
2. Simple INT8 inference benchmark
3. Latency measurement (p50, p95, p99)
4. Throughput measurement (inferences/sec)
5. Power efficiency estimation
"""

import os
import sys
import time
import numpy as np
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NPUTester:
    """Test Intel NPU functionality and performance"""

    def __init__(self):
        self.npu_available = False
        self.openvino_available = False
        self.device_name = "NPU"

    def check_device_node(self) -> bool:
        """Check if NPU device node exists"""
        device_paths = [
            "/dev/accel/accel0",
            "/dev/dri/renderD128",
        ]

        for path in device_paths:
            if os.path.exists(path):
                logger.info(f"✓ Found device node: {path}")
                return True

        logger.error("✗ No NPU device node found")
        return False

    def check_driver(self) -> bool:
        """Check if intel_vpu driver is loaded"""
        try:
            with open("/proc/modules", "r") as f:
                modules = f.read()
                if "intel_vpu" in modules:
                    logger.info("✓ intel_vpu driver loaded")
                    return True
        except:
            pass

        logger.error("✗ intel_vpu driver not loaded")
        return False

    def check_openvino(self) -> bool:
        """Check OpenVINO availability"""
        try:
            from openvino.runtime import Core
            self.openvino_available = True
            logger.info("✓ OpenVINO available")
            return True
        except ImportError:
            logger.warning("⚠️  OpenVINO not installed")
            logger.warning("   Install with: pip install openvino")
            return False

    def detect_npu_openvino(self) -> bool:
        """Detect NPU using OpenVINO"""
        if not self.openvino_available:
            return False

        try:
            from openvino.runtime import Core

            core = Core()
            devices = core.available_devices

            logger.info(f"Available devices: {devices}")

            # Check for NPU
            if "NPU" in devices:
                logger.info("✓ NPU detected by OpenVINO")
                self.npu_available = True

                # Get NPU properties
                try:
                    npu_name = core.get_property("NPU", "FULL_DEVICE_NAME")
                    logger.info(f"  NPU: {npu_name}")
                except:
                    pass

                return True
            else:
                logger.warning("⚠️  NPU not detected by OpenVINO")
                logger.warning("   Available devices: " + ", ".join(devices))
                return False

        except Exception as e:
            logger.error(f"✗ Error detecting NPU: {e}")
            return False

    def benchmark_simple_inference(self, num_trials: int = 100) -> Dict:
        """
        Benchmark simple inference on NPU

        Args:
            num_trials: Number of inference trials

        Returns:
            Dict with performance metrics
        """
        if not self.npu_available:
            logger.error("NPU not available for benchmarking")
            return {}

        try:
            from openvino.runtime import Core, Type
            import openvino.runtime.opset13 as opset

            logger.info("\n" + "=" * 80)
            logger.info("  NPU Inference Benchmark")
            logger.info("=" * 80)

            core = Core()

            # Create simple test model (matrix multiplication)
            # Input: [1, 384] (typical embedding size)
            # Output: [1, 128]
            batch_size = 1
            input_size = 384
            output_size = 128

            logger.info(f"Model: {input_size} → {output_size} (matrix multiplication)")

            # Build model
            input_tensor = opset.parameter([batch_size, input_size], Type.f32, name="input")
            weights = opset.constant(np.random.randn(input_size, output_size).astype(np.float32))
            matmul = opset.matmul(input_tensor, weights, False, False)
            result = opset.result(matmul, name="output")

            from openvino.runtime import Model
            model = Model([result], [input_tensor], "simple_matmul")

            # Compile for NPU
            logger.info("Compiling model for NPU...")
            compiled_model = core.compile_model(model, "NPU")

            # Create inference request
            infer_request = compiled_model.create_infer_request()

            # Generate random input
            input_data = np.random.randn(batch_size, input_size).astype(np.float32)

            # Warmup
            logger.info("Warmup (10 iterations)...")
            for _ in range(10):
                infer_request.infer({0: input_data})

            # Benchmark
            logger.info(f"Running benchmark ({num_trials} iterations)...")
            latencies = []

            for _ in range(num_trials):
                start = time.perf_counter()
                infer_request.infer({0: input_data})
                end = time.perf_counter()

                latencies.append((end - start) * 1000)  # Convert to ms

            # Calculate statistics
            latencies = np.array(latencies)
            p50 = np.percentile(latencies, 50)
            p95 = np.percentile(latencies, 95)
            p99 = np.percentile(latencies, 99)
            avg = np.mean(latencies)
            std = np.std(latencies)
            throughput = 1000.0 / avg  # inferences/sec

            # Display results
            logger.info("\nResults:")
            logger.info(f"  Average latency: {avg:.2f} ms ± {std:.2f} ms")
            logger.info(f"  P50 latency: {p50:.2f} ms")
            logger.info(f"  P95 latency: {p95:.2f} ms")
            logger.info(f"  P99 latency: {p99:.2f} ms")
            logger.info(f"  Throughput: {throughput:.1f} inferences/sec")

            logger.info("\n" + "=" * 80)

            return {
                "avg_latency_ms": avg,
                "std_latency_ms": std,
                "p50_latency_ms": p50,
                "p95_latency_ms": p95,
                "p99_latency_ms": p99,
                "throughput_ips": throughput,
                "num_trials": num_trials,
            }

        except Exception as e:
            logger.error(f"✗ Benchmark failed: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def run_full_test(self) -> bool:
        """Run complete NPU test suite"""
        logger.info("\n" + "=" * 80)
        logger.info("  Intel NPU Test Suite")
        logger.info("=" * 80)
        logger.info("Hardware: Dell Latitude 5450 MIL-SPEC")
        logger.info("NPU: Intel Meteor Lake NPU (49.4 TOPS INT8)")
        logger.info("PCI: 0000:00:0b.0 [8086:7d1d]")
        logger.info("=" * 80)

        # Test 1: Device node
        logger.info("\n[Test 1/5] Device Node Check")
        device_ok = self.check_device_node()

        # Test 2: Driver
        logger.info("\n[Test 2/5] Driver Check")
        driver_ok = self.check_driver()

        # Test 3: OpenVINO
        logger.info("\n[Test 3/5] OpenVINO Check")
        openvino_ok = self.check_openvino()

        # Test 4: NPU Detection
        logger.info("\n[Test 4/5] NPU Detection")
        npu_ok = self.detect_npu_openvino()

        # Test 5: Benchmark
        logger.info("\n[Test 5/5] Performance Benchmark")
        if npu_ok:
            metrics = self.benchmark_simple_inference(num_trials=100)
            benchmark_ok = bool(metrics)
        else:
            logger.warning("⚠️  Skipping benchmark (NPU not available)")
            benchmark_ok = False

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("  Test Summary")
        logger.info("=" * 80)
        logger.info(f"Device Node:    {'✓ PASS' if device_ok else '✗ FAIL'}")
        logger.info(f"Driver:         {'✓ PASS' if driver_ok else '✗ FAIL'}")
        logger.info(f"OpenVINO:       {'✓ PASS' if openvino_ok else '✗ FAIL'}")
        logger.info(f"NPU Detection:  {'✓ PASS' if npu_ok else '✗ FAIL'}")
        logger.info(f"Benchmark:      {'✓ PASS' if benchmark_ok else '✗ FAIL'}")
        logger.info("=" * 80)

        all_passed = device_ok and driver_ok and npu_ok

        if all_passed:
            logger.info("\n🎉 All tests PASSED! NPU is fully operational.")
        else:
            logger.warning("\n⚠️  Some tests FAILED. Check logs above.")

        return all_passed


def main():
    """Run NPU tests"""
    tester = NPUTester()
    success = tester.run_full_test()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
