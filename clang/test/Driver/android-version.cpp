// Check that we get the right Android version.

// RUN: not %clang -target aarch64-linux-androidS -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-ERROR %s

// CHECK-ERROR: error: Version S in triple aarch64-unknown-linux-android is invalid.

// RUN: not %clang -target armv7-linux-androideabiS -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-ERROR1 %s

// CHECK-ERROR1: error: Version S in triple armv7-unknown-linux-android is invalid.

// RUN: %clang -target aarch64-linux-android31 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-TARGET %s

// CHECK-TARGET: "aarch64-unknown-linux-android31"
