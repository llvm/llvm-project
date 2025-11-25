#!/usr/bin/env python3
"""
Interactive Hardware Detection Probe

Detects all available hardware:
- Intel NPU (Military-grade: 34-49.4 TOPS)
- Intel GNA (Gaussian Neural Accelerator)
- Intel Arc GPU
- Intel NCS2 sticks (2-3 units)
- AVX-512 on P-cores (CPUs 0-5)

Usage:
    python 03_test_hardware.py
"""

import os
import subprocess


def check_avx512():
    """Check AVX-512 availability"""
    print("\n" + "=" * 80)
    print("  AVX-512 DETECTION")
    print("=" * 80)

    if os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo", "r") as f:
            cpuinfo = f.read()

            if "avx512" in cpuinfo.lower():
                print("\n✓ AVX-512 AVAILABLE")

                # Extract AVX-512 features
                flags = []
                for line in cpuinfo.split("\n"):
                    if "flags" in line.lower():
                        flags = [f for f in line.split() if "avx512" in f.lower()]
                        break

                print(f"\nAVX-512 Features ({len(flags)}):")
                for i, flag in enumerate(sorted(set(flags)), 1):
                    print(f"  {i}. {flag}")

                print("\nP-cores (AVX-512 capable): 0-5")
                print("E-cores (no AVX-512): 6-15")

                # Check if E-cores are disabled
                e_core_online = []
                for cpu in range(6, 16):
                    path = f"/sys/devices/system/cpu/cpu{cpu}/online"
                    if os.path.exists(path):
                        with open(path, "r") as f:
                            if f.read().strip() == "1":
                                e_core_online.append(cpu)

                if e_core_online:
                    print(f"\n⚠ E-cores active: {len(e_core_online)}")
                    print("  Use task pinning: taskset -c 0-5 <program>")
                else:
                    print("\n✓ E-cores disabled (traditional AVX-512 unlock)")

            else:
                print("\n✗ AVX-512 NOT AVAILABLE")
                print("  Run: sudo ./avx512-unlock/unlock_avx512_advanced.sh enable")
    else:
        print("\n✗ /proc/cpuinfo not found")


def check_npu():
    """Check Intel NPU"""
    print("\n" + "=" * 80)
    print("  INTEL NPU DETECTION")
    print("=" * 80)

    try:
        import openvino as ov
        core = ov.Core()
        devices = core.available_devices

        if "NPU" in devices:
            print("\n✓ Intel NPU AVAILABLE")
            print("  Type: Military-grade (34-49.4 TOPS)")
            print("  Best for: INT8 quantized models")
            print("  Expected speedup: 3-4x vs CPU")

            # Get NPU properties
            try:
                props = core.get_property("NPU", "AVAILABLE_DEVICES")
                print(f"  Devices: {props}")
            except:
                pass
        else:
            print("\n✗ Intel NPU NOT DETECTED")
            print("  Available devices:", ", ".join(devices))

    except ImportError:
        print("\n✗ OpenVINO not installed")
        print("  Install: pip install openvino")


def check_arc_gpu():
    """Check Intel Arc GPU"""
    print("\n" + "=" * 80)
    print("  INTEL ARC GPU DETECTION")
    print("=" * 80)

    # Check via lspci
    try:
        result = subprocess.run(["lspci"], capture_output=True, text=True)
        lspci_output = result.stdout.lower()

        if "arc" in lspci_output or "dg2" in lspci_output:
            print("\n✓ Intel Arc GPU DETECTED")

            # Extract GPU info
            for line in lspci_output.split("\n"):
                if "vga" in line or "display" in line or "arc" in line:
                    print(f"  {line.strip()}")

            print("\n  Best for: FP16/INT8 inference")
            print("  Supports: OpenVINO, Level Zero, SYCL")

        else:
            print("\n✗ Intel Arc GPU NOT DETECTED")

    except FileNotFoundError:
        print("\n✗ lspci not available")


def check_ncs2():
    """Check NCS2 sticks"""
    print("\n" + "=" * 80)
    print("  INTEL NCS2 DETECTION")
    print("=" * 80)

    # Check via lsusb
    try:
        result = subprocess.run(["lsusb"], capture_output=True, text=True)
        lsusb_output = result.stdout

        ncs2_count = lsusb_output.lower().count("myriad")

        if ncs2_count > 0:
            print(f"\n✓ Intel NCS2 DETECTED ({ncs2_count} units)")
            print(f"  Total inference capacity: ~{ncs2_count * 10} tokens/sec")
            print("  Best for: Edge deployment, parallel inference")
            print("  Supports: OpenVINO IR format (INT8)")

            for line in lsusb_output.split("\n"):
                if "myriad" in line.lower():
                    print(f"  {line.strip()}")
        else:
            print("\n✗ Intel NCS2 NOT DETECTED")

    except FileNotFoundError:
        print("\n✗ lsusb not available")


def check_gna():
    """Check Gaussian Neural Accelerator"""
    print("\n" + "=" * 80)
    print("  INTEL GNA DETECTION")
    print("=" * 80)

    try:
        import openvino as ov
        core = ov.Core()
        devices = core.available_devices

        if "GNA" in devices:
            print("\n✓ Intel GNA AVAILABLE")
            print("  Type: Gaussian Neural Accelerator 3.5")
            print("  Best for: Low-power audio/signal processing")
            print("  Used by: Audio preprocessing, security monitoring")
        else:
            print("\n✗ Intel GNA NOT DETECTED")

    except ImportError:
        print("\n✗ OpenVINO not installed")


def check_cuda():
    """Check CUDA GPU (if any)"""
    print("\n" + "=" * 80)
    print("  NVIDIA CUDA DETECTION")
    print("=" * 80)

    try:
        import torch
        if torch.cuda.is_available():
            print("\n✓ CUDA GPU AVAILABLE")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("\n✗ CUDA GPU NOT AVAILABLE")
    except ImportError:
        print("\n✗ PyTorch not installed")


def hardware_summary():
    """Show hardware summary"""
    print("\n" + "=" * 80)
    print("  HARDWARE SUMMARY")
    print("=" * 80)

    print("\nDell Latitude 5450 MIL-SPEC Hardware Configuration:")
    print("\n  CPU:")
    print("    • Intel Core Ultra 7 165H (Meteor Lake)")
    print("    • 6 P-cores (AVX-512 capable)")
    print("    • 10 E-cores")

    print("\n  AI Accelerators:")
    print("    • Intel NPU (Military: 34-49.4 TOPS)")
    print("    • Intel GNA 3.5")
    print("    • Intel Arc GPU")
    print("    • Intel NCS2 sticks (2-3 units)")

    print("\n  Recommended Quantization:")
    print("    • NPU: INT8 (best throughput)")
    print("    • Arc GPU: FP16/INT8 (balanced)")
    print("    • P-cores + AVX-512: BF16 (best quality)")
    print("    • NCS2: INT8 OpenVINO IR (edge deployment)")

    print("\n  Total AI Compute:")
    print("    • NPU: 34-49 TOPS (INT8)")
    print("    • Arc GPU: ~8-16 TFLOPS (FP16)")
    print("    • NCS2: ~3-6 TOPS (2-3 units)")
    print("    • Total: ~45-70 TOPS equivalent")


def main():
    """Run all hardware checks"""
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║          HARDWARE DETECTION INTERACTIVE PROBE                            ║
║          Dell Latitude 5450 MIL-SPEC Edition                             ║
╚══════════════════════════════════════════════════════════════════════════╝
""")

    while True:
        print("\n" + "=" * 80)
        print("  HARDWARE DETECTION MENU")
        print("=" * 80)
        print("\n1. Check AVX-512")
        print("2. Check Intel NPU")
        print("3. Check Intel Arc GPU")
        print("4. Check Intel NCS2 sticks")
        print("5. Check Intel GNA")
        print("6. Check NVIDIA CUDA (if any)")
        print("7. Run all checks")
        print("8. Show hardware summary")
        print("0. Exit")

        choice = input("\nSelect option: ").strip()

        if choice == "1":
            check_avx512()
        elif choice == "2":
            check_npu()
        elif choice == "3":
            check_arc_gpu()
        elif choice == "4":
            check_ncs2()
        elif choice == "5":
            check_gna()
        elif choice == "6":
            check_cuda()
        elif choice == "7":
            check_avx512()
            check_npu()
            check_arc_gpu()
            check_ncs2()
            check_gna()
            check_cuda()
        elif choice == "8":
            hardware_summary()
        elif choice == "0":
            print("\nExiting...")
            break
        else:
            print("\nInvalid option!")


if __name__ == "__main__":
    main()
