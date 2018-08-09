// Check that -stack-protector doesn't get passed down to device-side
// compilation.
//
// REQUIRES: clang-driver
//
// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_20 \
// RUN:   -fstack-protector-all %s 2>&1 | \
// RUN: FileCheck %s
//
// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_20 \
// RUN:   -fstack-protector-strong %s 2>&1 | \
// RUN: FileCheck %s
//
// RUN: %clang -### -target x86_64-linux-gnu -c --cuda-gpu-arch=sm_20 \
// RUN:   -fstack-protector %s 2>&1 | \
// RUN: FileCheck %s
//
// CHECK-NOT: error: unsupported option '-fstack-protector
// CHECK-DAG: "-fcuda-is-device"
// CHECK-NOT: "-stack-protector"
// CHECK-NOT: "-stack-protector-buffer-size"
// CHECK-DAG: "-triple" "x86_64-unknown-linux-gnu"
// CHECK: "-stack-protector"
