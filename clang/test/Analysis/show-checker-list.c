// RUN: %clang_cc1 -analyzer-checker-help \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK

// RUN: %clang_cc1 -analyzer-checker-help-hidden \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-HIDDEN

// CHECK: core.DivideZero
// CHECK-HIDDEN: core.DivideZero

// CHECK-NOT: unix.DynamicMemoryModeling
// CHECK-HIDDEN: unix.DynamicMemoryModeling
