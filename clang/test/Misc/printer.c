// RUN: %clang_cc1 -emit-llvm -fexperimental-new-pass-manager -mllvm -print-before-all %s -o %t 2>&1 | FileCheck %s --check-prefix=CHECK-BEFORE
// RUN: %clang_cc1 -emit-llvm -fexperimental-new-pass-manager -mllvm -print-after-all %s -o %t 2>&1 | FileCheck %s --check-prefix=CHECK-AFTER
// CHECK-BEFORE: *** IR Dump Before AlwaysInlinerPass
// CHECK-AFTER: *** IR Dump After AlwaysInlinerPass
void foo(void) {}
