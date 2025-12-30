// Test that default features (e.g. flagm/sb/ssbs for 8.5) can be disabled via -march.

// RUN: %clang --target=aarch64 -march=armv8.5-a+noflagm+nosb+nossbs -c %s -### 2>&1 | FileCheck %s
// CHECK: "-triple" "aarch64"
// CHECK-SAME: "-target-feature" "+v8.5a"
// CHECK-SAME: "-target-feature" "-flagm"
// CHECK-SAME: "-target-feature" "-sb"
// CHECK-SAME: "-target-feature" "-ssbs"

// CHECK-NOT: "-target-feature" "+flagm"
// CHECK-NOT: "-target-feature" "+sb"
// CHECK-NOT: "-target-feature" "+ssbs"
