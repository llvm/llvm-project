// RUN: %clang -MD -no-canonical-prefixes -### %s 2>&1 | FileCheck %s --check-prefix=CHECK-NO
// RUN: %clang -MD -canonical-prefixes -### %s 2>&1 | FileCheck %s --check-prefix=CHECK-YES
// RUN: %clang -MD -### %s 2>&1 | FileCheck %s --check-prefix=CHECK-YES

// CHECK-YES: "-canonical-system-headers"
// CHECK-NO-NOT: "-canonical-system-headers"
