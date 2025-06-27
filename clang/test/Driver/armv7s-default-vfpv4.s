/// Ensure that we can assemble VFPv4 by just specifying an armv7s target.

// REQUIRES: arm-registered-target
// RUN: %clang -c -target armv7s-apple-darwin -o /dev/null %s 2>&1 | FileCheck --check-prefix=CHECK-STDERR %s --allow-empty
// RUN: %clang -c -target armv7s-apple-darwin -o /dev/null %s -### 2>&1 | FileCheck --check-prefix=CHECK-TARGET-FEATURES %s

/// Check that no errors or warnings are present when assembling using cc1as.
// CHECK-STDERR-NOT: error:
// CHECK-STDERR-NOT: warning:

// CHECK-TARGET-FEATURES: "-target-feature" "+vfp4"

vfma.f32 q1, q2, q3
