// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// Make sure the appropriate device specific library is available.

// We don't include every target in the test directory, so just pick a valid
// target not included in the test.

// RUN: %clang -### -v -target x86_64-linux-gnu --cuda-gpu-arch=gfx902 \
// RUN:   --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=COMMON,GFX902-DEFAULTLIBS %s

// Should not interpret -nostdlib as disabling offload libraries.
// RUN: %clang -### -v -target x86_64-linux-gnu --cuda-gpu-arch=gfx902 -nostdlib \
// RUN:   --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=COMMON,GFX902-DEFAULTLIBS %s


// RUN: %clang -### -v -target x86_64-linux-gnu --cuda-gpu-arch=gfx902 -nogpulib \
// RUN:   --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=COMMON,NODEFAULTLIBS %s


// GFX902-DEFAULTLIBS: error: cannot find ROCm device library for gfx902. Provide its path via --rocm-path or --rocm-device-lib-path, or pass -nogpulib to build without ROCm device library

// NODEFAULTLIBS-NOT: error: cannot find

// COMMON: "-triple" "amdgcn-amd-amdhsa"
