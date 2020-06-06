// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// Make sure the appropriate device specific library is available.

// We don't include every target in the test directory, so just pick a valid
// target not included in the test.

// RUN: %clang -### -v -target x86_64-linux-gnu --cuda-gpu-arch=gfx902 \
// RUN:   --rocm-path=%S/Inputs/rocm-device-libs %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=COMMON,GFX902-DEFAULTLIBS %s

// Should not interpret -nostdlib as disabling offload libraries.
// RUN: %clang -### -v -target x86_64-linux-gnu --cuda-gpu-arch=gfx902 -nostdlib \
// RUN:   --rocm-path=%S/Inputs/rocm-device-libs %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=COMMON,GFX902-DEFAULTLIBS %s


// RUN: %clang -### -v -target x86_64-linux-gnu --cuda-gpu-arch=gfx902 -nogpulib \
// RUN:   --rocm-path=%S/Inputs/rocm-device-libs %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=COMMON,GFX902,NODEFAULTLIBS %s


// GFX902-DEFAULTLIBS: error: cannot find device library for gfx902. Provide path to different ROCm installation via --rocm-path, or pass -nogpulib to build without linking default libraries.

// NODEFAULTLIBS-NOT: error: cannot find
