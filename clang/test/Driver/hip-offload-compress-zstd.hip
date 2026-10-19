// REQUIRES: zstd

// Test HIP offload compression.

// RUN: rm -rf %t.bc
// RUN: %clang -### -v --target=x86_64-linux-gnu \
// RUN:   -x hip --offload-arch=gfx1100 --offload-arch=gfx1101 \
// RUN:   -fgpu-rdc -nogpuinc -nogpulib \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu \
// RUN:   --offload-compress --offload-compression-level=9 \
// RUN:   --gpu-bundle-output \
// RUN:   -o %t.bc \
// RUN: 2>&1 | FileCheck %s --check-prefix=NEWDRIVER

// NEWDRIVER: clang-linker-wrapper{{.*}}"--compress" "--verbose" "--compression-level=9"
