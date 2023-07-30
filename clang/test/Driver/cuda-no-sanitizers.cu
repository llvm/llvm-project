// Check that -fsanitize=foo doesn't get passed down to device-side
// compilation.
//
//
// RUN: not %clang -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=sm_20 -fsanitize=address %s 2>&1 | \
// RUN:   FileCheck %s

// CHECK-NOT: error: unsupported option '-fsanitize=address'
// CHECK-DAG: "-fcuda-is-device"
// CHECK-NOT: "-fsanitize=address"
// CHECK-DAG: "-triple" "x86_64-unknown-linux-gnu"
// CHECK: "-fsanitize=address"
