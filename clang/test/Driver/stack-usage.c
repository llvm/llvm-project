// RUN: %clang -target aarch64-unknown %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ABSENT
// CHECK-ABSENT-NOT: "-stack-usage-file"

// RUN: %clang -target aarch64-unknown -fstack-usage %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-PRESENT
// CHECK-PRESENT: "-stack-usage-file"

int foo() { return 42; }
