#!/usr/bin/env python3
"""
Hardware Accelerator Benchmark Tool
====================================
Comprehensive benchmarking for all hardware accelerators.

Performance Targets:
- NCS2: 10 TOPS per device (30 TOPS for 3 devices)
- NPU: 30 TOPS optimized
- Military NPU: 100+ TOPS
- GPU: 100+ TOPS per device

Total Target: 150+ TOPS

Author: LAT5150DRVMIL AI Platform
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "02-ai-engine"))

import numpy as np

from hardware_config import print_hardware_summary
from unified_accelerator import (
    AcceleratorType,
    get_unified_manager
)


class Colors:
    """Terminal colors."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text: str):
    """Print formatted header."""
    print()
    print(Colors.HEADER + Colors.BOLD + "=" * 80 + Colors.ENDC)
    print(Colors.HEADER + Colors.BOLD + f"  {text}" + Colors.ENDC)
    print(Colors.HEADER + Colors.BOLD + "=" * 80 + Colors.ENDC)
    print()


def print_success(text: str):
    """Print success message."""
    print(Colors.OKGREEN + "✓ " + text + Colors.ENDC)


def print_warning(text: str):
    """Print warning message."""
    print(Colors.WARNING + "⚠ " + text + Colors.ENDC)


def print_fail(text: str):
    """Print failure message."""
    print(Colors.FAIL + "✗ " + text + Colors.ENDC)


def print_info(text: str):
    """Print info message."""
    print(Colors.OKCYAN + "ℹ " + text + Colors.ENDC)


def benchmark_accelerator(
    manager,
    accelerator_type: AcceleratorType,
    duration_seconds: int = 10,
    batch_size: int = 8
) -> dict:
    """
    Benchmark a specific accelerator.

    Args:
        manager: Unified accelerator manager
        accelerator_type: Type of accelerator to benchmark
        duration_seconds: Benchmark duration
        batch_size: Batch size for inference

    Returns:
        Benchmark results dictionary
    """
    print_info(f"Benchmarking {accelerator_type.value.upper()}...")

    # Create test input
    input_data = np.random.rand(batch_size, 3, 224, 224).astype(np.float32)

    # Statistics
    total_inferences = 0
    latencies = []
    start_time = time.time()

    # Warm-up
    print("  Warming up...")
    for _ in range(10):
        manager.submit_inference(
            model_id="benchmark_model",
            input_data=input_data,
            preferred_accelerator=accelerator_type
        )

    time.sleep(1)

    # Benchmark
    print(f"  Running benchmark for {duration_seconds} seconds...")

    while time.time() - start_time < duration_seconds:
        req_start = time.time()

        manager.submit_inference(
            model_id="benchmark_model",
            input_data=input_data,
            preferred_accelerator=accelerator_type
        )

        total_inferences += 1

        # Track latency (approximate)
        latencies.append((time.time() - req_start) * 1000)

        # Don't overload
        time.sleep(0.001)

    elapsed = time.time() - start_time

    # Calculate metrics
    if total_inferences > 0 and latencies:
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        throughput_fps = total_inferences / elapsed

        # Estimate TOPS (assume 1M ops per inference)
        ops_per_inference = 1_000_000
        ops_per_sec = throughput_fps * ops_per_inference
        achieved_tops = ops_per_sec / 1e12

        return {
            "accelerator": accelerator_type.value,
            "success": True,
            "total_inferences": total_inferences,
            "duration_seconds": elapsed,
            "avg_latency_ms": avg_latency,
            "min_latency_ms": min_latency,
            "max_latency_ms": max_latency,
            "throughput_fps": throughput_fps,
            "achieved_tops": achieved_tops
        }
    else:
        return {
            "accelerator": accelerator_type.value,
            "success": False,
            "error": "No inferences completed"
        }


def print_results(results: dict):
    """Print benchmark results."""
    print()
    print(Colors.BOLD + f"Accelerator: {results['accelerator'].upper()}" + Colors.ENDC)
    print("-" * 80)

    if not results.get("success", False):
        print_fail(f"Benchmark failed: {results.get('error', 'Unknown error')}")
        return

    print(f"  Total Inferences:  {results['total_inferences']:,}")
    print(f"  Duration:          {results['duration_seconds']:.2f} seconds")
    print(f"  Throughput:        {results['throughput_fps']:.1f} FPS")
    print(f"  Achieved TOPS:     {results['achieved_tops']:.2f}")
    print()
    print(f"  Latency:")
    print(f"    Average:         {results['avg_latency_ms']:.2f} ms")
    print(f"    Min:             {results['min_latency_ms']:.2f} ms")
    print(f"    Max:             {results['max_latency_ms']:.2f} ms")
    print()

    # Check targets
    targets = {
        "ncs2": 10.0,  # 10 TOPS per device (30 total for 3)
        "npu": 30.0,  # 30 TOPS optimized
        "military_npu": 100.0,  # 100 TOPS
        "cuda": 100.0,  # 100 TOPS per GPU
    }

    target = targets.get(results['accelerator'], 0.0)

    if target > 0:
        achievement = (results['achieved_tops'] / target) * 100

        if achievement >= 90:
            print_success(f"TARGET ACHIEVED: {achievement:.1f}% of {target} TOPS target")
        elif achievement >= 70:
            print_warning(f"CLOSE TO TARGET: {achievement:.1f}% of {target} TOPS target")
        else:
            print_fail(f"BELOW TARGET: {achievement:.1f}% of {target} TOPS target")


def run_comprehensive_benchmark(
    duration: int = 10,
    output_file: str = None
) -> dict:
    """
    Run comprehensive benchmark of all accelerators.

    Args:
        duration: Benchmark duration per accelerator
        output_file: Optional output JSON file

    Returns:
        Comprehensive results dictionary
    """
    print_header("Hardware Accelerator Comprehensive Benchmark")

    # Print hardware summary
    print_hardware_summary()

    # Initialize unified manager
    print_info("Initializing unified accelerator manager...")
    manager = get_unified_manager()

    print()
    print(f"Total Available TOPS: {Colors.BOLD}{manager.get_total_tops():.1f}{Colors.ENDC}")
    print()

    # Benchmark each accelerator
    all_results = {
        "timestamp": time.time(),
        "total_tops": manager.get_total_tops(),
        "benchmarks": []
    }

    for accel_type, cap in manager.accelerators.items():
        if cap.is_available:
            print()
            result = benchmark_accelerator(manager, accel_type, duration)
            print_results(result)
            all_results["benchmarks"].append(result)

    # Summary
    print_header("Benchmark Summary")

    total_achieved_tops = sum(
        r.get("achieved_tops", 0.0)
        for r in all_results["benchmarks"]
        if r.get("success", False)
    )

    target_tops = 150.0  # Target: 30 NCS2 + 30 NPU + 100 Military + more

    print(f"Total Achieved TOPS:    {Colors.BOLD}{total_achieved_tops:.2f}{Colors.ENDC}")
    print(f"Target TOPS:            {Colors.BOLD}{target_tops:.2f}{Colors.ENDC}")
    print(f"Achievement:            {Colors.BOLD}{(total_achieved_tops/target_tops)*100:.1f}%{Colors.ENDC}")
    print()

    # Per-accelerator summary
    print("Accelerator Performance:")
    print("-" * 80)

    for result in all_results["benchmarks"]:
        if result.get("success", False):
            accel = result['accelerator'].upper()
            tops = result['achieved_tops']
            fps = result['throughput_fps']
            latency = result['avg_latency_ms']

            print(f"  {accel:15s}  {tops:6.2f} TOPS  {fps:8.1f} FPS  {latency:6.2f}ms")

    print()

    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print_success(f"Results saved to: {output_file}")

    return all_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark hardware accelerators",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full benchmark (10 seconds per accelerator)
  ./benchmark-accelerators.py

  # Run longer benchmark with output file
  ./benchmark-accelerators.py --duration 30 --output results.json

  # Quick benchmark
  ./benchmark-accelerators.py --duration 5
        """
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=10,
        help="Benchmark duration per accelerator (seconds)"
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output file for results (JSON)"
    )

    args = parser.parse_args()

    # Run benchmark
    try:
        results = run_comprehensive_benchmark(
            duration=args.duration,
            output_file=args.output
        )

        # Exit code based on achievement
        total_achieved = sum(
            r.get("achieved_tops", 0.0)
            for r in results["benchmarks"]
            if r.get("success", False)
        )

        if total_achieved >= 135:  # 90% of 150 TOPS target
            print()
            print_success("🎉 EXCELLENT: All performance targets achieved!")
            return 0
        elif total_achieved >= 105:  # 70% of 150 TOPS target
            print()
            print_warning("⚠️  GOOD: Close to performance targets")
            return 0
        else:
            print()
            print_fail("❌ NEEDS IMPROVEMENT: Review optimization steps")
            return 1

    except KeyboardInterrupt:
        print()
        print_warning("Benchmark interrupted by user")
        return 130
    except Exception as e:
        print()
        print_fail(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
