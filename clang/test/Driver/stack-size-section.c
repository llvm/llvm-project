// RUN: %clang -target x86_64-unknown %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ABSENT
// RUN: %clang -target x86_64-scei-ps4 -fno-stack-size-section %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ABSENT
// CHECK-ABSENT-NOT: -fstack-size-section

// RUN: %clang -target x86_64-unknown -fstack-size-section -### 2>&1 | FileCheck %s --check-prefix=CHECK-PRESENT
// RUN: %clang -target x86_64-scei-ps4 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-PRESENT
// CHECK-PRESENT: -fstack-size-section

int foo() { return 42; }
