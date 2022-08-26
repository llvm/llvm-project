// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: %clang -ccc-print-phases --target=x86_64-linux-gnu \
// RUN:   -x hip --cuda-gpu-arch=gfx803 -c \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip 2>&1 \
// RUN: | FileCheck %s

// RUN: not %clang -ccc-print-phases --target=x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx803 -c \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip 2>&1 \
// RUN: | FileCheck -check-prefix=MIX %s

// RUN: not %clang -ccc-print-phases --target=x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx803 -c \
// RUN:   --hip-link %S/Inputs/hip_multiple_inputs/a.cu 2>&1 \
// RUN: | FileCheck -check-prefix=MIX %s

// CHECK-NOT: error: mixed CUDA and HIP compilation is not supported
// MIX: error: mixed CUDA and HIP compilation is not supported
