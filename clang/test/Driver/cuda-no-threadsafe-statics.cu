// Check that -fno-thread-safe-statics get passed down to device-side
// compilation only.
//
// RUN: not %clang -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=sm_20 %s 2>&1 \
// RUN:   | FileCheck %s
//
// CHECK: "-fcuda-is-device"
// CHECK-SAME: "-fno-threadsafe-statics"
// CHECK: "-triple" "x86_64-unknown-linux-gnu"
// CHECK-NOT: "-fno-threadsafe-statics"
