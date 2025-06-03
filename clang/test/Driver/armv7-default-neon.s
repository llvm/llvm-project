/// Ensure that we can assemble NEON by just specifying an armv7
/// Apple or Windows target.

// REQUIRES: arm-registered-target
// RUN: %clang -c -target armv7-apple-darwin -o /dev/null %s 2>&1 | FileCheck --check-prefix=CHECK-STDERR %s --allow-empty
// RUN: %clang -c -target armv7-apple-darwin -o /dev/null %s -### 2>&1 | FileCheck --check-prefix=CHECK-TARGET-FEATURES %s
// RUN: %clang -c -target armv7-windows -o /dev/null %s 2>&1 | FileCheck --check-prefix=CHECK-STDERR %s --allow-empty
// RUN: %clang -c -target armv7-windows -o /dev/null %s -### 2>&1 | FileCheck --check-prefix=CHECK-TARGET-FEATURES %s

/// Check that no errors or warnings are present when assembling using cc1as.
// CHECK-STDERR-NOT: error:
// CHECK-STDERR-NOT: warning:

// CHECK-TARGET-FEATURES: "-target-feature" "+neon"

vadd.i32 q0, q0, q0
