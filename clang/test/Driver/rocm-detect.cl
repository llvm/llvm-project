// REQUIRES: clang-driver
// REQUIRES: amdgpu-registered-target

// Make sure the appropriate device specific library is available.

// We don't include every target in the test directory, so just pick a valid
// target not included in the test.

// RUN: %clang -### -v -target amdgcn-amd-amdhsa -mcpu=gfx902 \
// RUN:   --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=COMMON,GFX902-DEFAULTLIBS %s


// RUN: %clang -### -v -target amdgcn-amd-amdhsa -mcpu=gfx902 -nogpulib \
// RUN:   --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=COMMON,NODEFAULTLIBS %s

// GFX902-DEFAULTLIBS: error: cannot find ROCm device library for gfx902; provide its path via '--rocm-path' or '--rocm-device-lib-path', or pass '-nogpulib' to build without ROCm device library

// NODEFAULTLIBS-NOT: error: cannot find

// COMMON: "-triple" "amdgcn-amd-amdhsa"
