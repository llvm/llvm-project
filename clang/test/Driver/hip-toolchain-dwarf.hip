// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: %clang -### -c -target x86_64-linux-gnu -fgpu-rdc \
// RUN:   -x hip --cuda-gpu-arch=gfx803 %s \
// RUN:   -Xarch_gfx803 -g 2>&1 | FileCheck %s -check-prefix=DWARF_VER

// DWARF_VER: "-dwarf-version=5"
