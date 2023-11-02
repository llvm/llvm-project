// REQUIRES: system-windows
// The default on Windows is false due to #70011.

// RUN: %clang -MD -no-canonical-prefixes -### %s 2>&1 | FileCheck %s --check-prefix=CHECK-NO
// RUN: %clang -MD -canonical-prefixes -### %s 2>&1 | FileCheck %s --check-prefix=CHECK-YES
// RUN: %clang -MD -### %s 2>&1 | FileCheck %s --check-prefix=CHECK-NO

// CHECK-YES: "-canonical-system-headers"
// CHECK-NO-NOT: "-canonical-system-headers"
