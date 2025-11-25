#!/usr/bin/env python3
"""
Direct SIMD testing for Meteor Lake
Simple and safe approach
"""

import subprocess
import os
import json
from pathlib import Path

def compile_and_test():
    """Compile and run the C test program"""
    
    print("Building AVX test binary...")
    
    # Create simple test C file
    test_c = """
#define _GNU_SOURCE
#include <stdio.h>
#include <sched.h>
#include <signal.h>
#include <setjmp.h>
#include <immintrin.h>

static jmp_buf jmpbuf;

void sigill_handler(int sig) {
    longjmp(jmpbuf, 1);
}

int test_avx512(void) {
    signal(SIGILL, sigill_handler);
    if (setjmp(jmpbuf) == 0) {
        // Try AVX-512 - will SIGILL if not supported
        asm volatile("vpxord %%zmm0, %%zmm0, %%zmm0" ::: "zmm0");
        return 1;
    }
    return 0;
}

int test_avx2(void) {
    signal(SIGILL, sigill_handler);
    if (setjmp(jmpbuf) == 0) {
        // Try AVX2
        asm volatile("vpxor %%ymm0, %%ymm0, %%ymm0" ::: "ymm0");
        return 1;
    }
    return 0;
}

int main() {
    cpu_set_t cpuset;
    
    // Test AVX-512 on P-cores
    printf("Testing P-cores (0-11) for AVX-512:\\n");
    int avx512_count = 0;
    for (int i = 0; i < 12; i++) {
        CPU_ZERO(&cpuset);
        CPU_SET(i, &cpuset);
        if (sched_setaffinity(0, sizeof(cpuset), &cpuset) == 0) {
            if (test_avx512()) {
                printf("  Core %d: AVX-512 YES\\n", i);
                avx512_count++;
            } else {
                printf("  Core %d: AVX-512 NO\\n", i);
            }
        }
    }
    
    // Test AVX2
    printf("\\nTesting AVX2: ");
    if (test_avx2()) {
        printf("YES\\n");
    } else {
        printf("NO\\n");
    }
    
    printf("\\nSummary: %d cores with AVX-512\\n", avx512_count);
    return 0;
}
"""
    
    # Write test file
    with open("/home/john/LAT5150DRVMIL/test_simd.c", "w") as f:
        f.write(test_c)
    
    # Compile with various AVX flags
    compile_cmd = [
        "gcc", "-O2", "-o", "/home/john/LAT5150DRVMIL/test_simd",
        "/home/john/LAT5150DRVMIL/test_simd.c",
        "-mavx2", "-mavx512f"
    ]
    
    try:
        result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            print(f"Compilation failed: {result.stderr}")
            return None
    except subprocess.TimeoutExpired:
        print("Compilation timed out")
        return None
    
    # Run the test
    print("\nRunning SIMD detection test...")
    try:
        result = subprocess.run(
            ["/home/john/LAT5150DRVMIL/test_simd"],
            capture_output=True,
            text=True,
            timeout=5
        )
        print(result.stdout)
        
        # Parse results
        avx512_capable = "AVX-512 YES" in result.stdout
        avx2_capable = "AVX2: YES" in result.stdout
        
        return {
            "avx512": avx512_capable,
            "avx2": avx2_capable,
            "recommended": "AVX-512" if avx512_capable else "AVX2" if avx2_capable else "SSE4.2"
        }
    except subprocess.TimeoutExpired:
        print("Test timed out - likely no AVX support")
        return {"avx512": False, "avx2": False, "recommended": "SSE4.2"}
    except Exception as e:
        print(f"Test failed: {e}")
        return None

def get_microcode_version():
    """Try to get microcode version"""
    try:
        # Try from sysfs
        path = Path("/sys/devices/system/cpu/cpu0/microcode/version")
        if path.exists():
            version = path.read_text().strip()
            if version.startswith("0x"):
                return int(version, 16)
            return int(version)
    except:
        pass
    
    # Try cpuid
    try:
        result = subprocess.run(
            ["grep", "microcode", "/proc/cpuinfo"],
            capture_output=True,
            text=True,
            timeout=1
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if lines:
                # Parse first microcode line
                parts = lines[0].split(':')
                if len(parts) == 2:
                    version = parts[1].strip()
                    if version.startswith("0x"):
                        return int(version, 16)
                    return int(version)
    except:
        pass
    
    return None

def main():
    print("=" * 60)
    print("Direct SIMD Testing for Intel Meteor Lake")
    print("Testing actual instructions, not trusting CPUID")
    print("=" * 60)
    
    # Get microcode
    microcode = get_microcode_version()
    if microcode:
        print(f"\nMicrocode version: 0x{microcode:x}")
        if microcode < 0x1c:
            print("⚠ WARNING: Microcode < 0x1c, AVX-512 may be unstable")
    else:
        print("\nMicrocode version: Unknown")
    
    # Test SIMD
    capabilities = compile_and_test()
    
    if capabilities:
        print("\n" + "=" * 60)
        print("RESULTS:")
        print(f"AVX-512 Support: {capabilities['avx512']}")
        print(f"AVX2 Support: {capabilities['avx2']}")
        print(f"Recommended: {capabilities['recommended']}")
        
        # Save results
        results = {
            "microcode": microcode,
            "avx512": capabilities["avx512"],
            "avx2": capabilities["avx2"],
            "recommended": capabilities["recommended"],
            "p_cores": list(range(12)),
            "e_cores": list(range(12, 22))
        }
        
        with open("/home/john/LAT5150DRVMIL/simd_capabilities.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n[+] Results saved to simd_capabilities.json")
        print("\n⚠ Remember: AVX-512 only works on P-cores (0-11)")
        print("⚠ E-cores (12-21) do NOT support AVX-512")
    else:
        print("\n[!] SIMD detection failed")

if __name__ == "__main__":
    main()