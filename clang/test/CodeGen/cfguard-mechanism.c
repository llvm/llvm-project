// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s -check-prefix=AUTOMATIC
// RUN: %clang_cc1 -fwin-cfg-mechanism=automatic -emit-llvm %s -o - | FileCheck %s -check-prefix=AUTOMATIC
// RUN: %clang_cc1 -fwin-cfg-mechanism=dispatch -emit-llvm %s -o - | FileCheck %s -check-prefix=DISPATCH
// RUN: %clang_cc1 -fwin-cfg-mechanism=check -emit-llvm %s -o - | FileCheck %s -check-prefix=CHECK
// RUN: %clang -fwin-cfg-mechanism=dispatch -S -emit-llvm %s -o - | FileCheck %s -check-prefix=DISPATCH

void f(void) {}

// CHECK: !"cfguard-mechanism", i32 1}
// DISPATCH: !"cfguard-mechanism", i32 2}
// AUTOMATIC-NOT: "cfguard-mechanism"
