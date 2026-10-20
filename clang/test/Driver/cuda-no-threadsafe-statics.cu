// Check that -fno-thread-safe-statics get passed down to device-side
// compilation only.
//
// RUN: %clang -### -x cuda --target=x86_64-linux-gnu -c --cuda-gpu-arch=sm_20 %s \
// RUN:            -nocudainc -nocudalib --cuda-path=%S/Inputs/CUDA_80/usr/local/cuda \
// RUN:            2>&1 | FileCheck %s

// RUN: %clang -### -x hip --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx1010 %s \
// RUN:            -nocudainc -nocudalib 2>&1 | FileCheck %s
//
// CHECK: "-fcuda-is-device"
// CHECK-SAME: "-fno-threadsafe-statics"
// CHECK: "-triple" "x86_64-unknown-linux-gnu"
// CHECK-NOT: "-fno-threadsafe-statics"
