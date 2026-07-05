// RUN: %clang -### -nogpuinc -nogpulib --target=x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx900 \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s

// CHECK-NOT: "-mlink-builtin-bitcode"
