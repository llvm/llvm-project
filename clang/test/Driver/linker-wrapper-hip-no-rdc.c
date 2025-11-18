// UNSUPPORTED: system-windows
// REQUIRES: amdgpu-registered-target
// REQUIRES: lld

// Test HIP non-RDC linker wrapper behavior with new offload driver.
// The linker wrapper should output .hipfb files directly without using -r option.

// An externally visible variable so static libraries extract.
__attribute__((visibility("protected"), used)) int x;

// Create device binaries and package them
// RUN: %clang -cc1 %s -triple amdgcn-amd-amdhsa -emit-llvm-bc -o %t.amdgpu.bc
// RUN: llvm-offload-binary -o %t.out \
// RUN:   --image=file=%t.amdgpu.bc,kind=hip,triple=amdgcn-amd-amdhsa,arch=gfx1100 \
// RUN:   --image=file=%t.amdgpu.bc,kind=hip,triple=amdgcn-amd-amdhsa,arch=gfx1200

// Test that linker wrapper outputs .hipfb file without -r option for HIP non-RDC
// The linker wrapper is called directly with the packaged device binary (not embedded in host object)
// Note: When called directly (not through the driver), the linker wrapper processes architectures
// from the packaged binary. The test verifies it can process at least one architecture correctly.
// RUN: clang-linker-wrapper --emit-fatbin-only --linker-path=/usr/bin/ld %t.out -o %t.hipfb 2>&1

// Verify the fat binary was created
// RUN: test -f %t.hipfb

// List code objects in the fat binary
// RUN: clang-offload-bundler -type=o -input=%t.hipfb -list | FileCheck %s --check-prefix=HIP-FATBIN-LIST

// HIP-FATBIN-LIST-DAG: hip-amdgcn-amd-amdhsa--gfx1100
// HIP-FATBIN-LIST-DAG: hip-amdgcn-amd-amdhsa--gfx1200
// HIP-FATBIN-LIST-DAG: host-x86_64-unknown-linux-gnu-

// Extract code objects for both architectures from the fat binary
// RUN: clang-offload-bundler -type=o -targets=hip-amdgcn-amd-amdhsa--gfx1100,hip-amdgcn-amd-amdhsa--gfx1200 \
// RUN:   -output=%t.gfx1100.co -output=%t.gfx1200.co -input=%t.hipfb -unbundle

// Verify extracted code objects exist and are not empty
// RUN: test -f %t.gfx1100.co
// RUN: test -s %t.gfx1100.co
// RUN: test -f %t.gfx1200.co
// RUN: test -s %t.gfx1200.co
