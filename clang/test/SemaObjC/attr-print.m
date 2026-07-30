// RUN: %clang_cc1 %s -fobjc-arc -ast-print | FileCheck %s

__strong id x;
id y;
__strong id z;

// CHECK: __strong id x;
// CHECK-NOT: __strong id y;
// CHECK: __strong id z;
