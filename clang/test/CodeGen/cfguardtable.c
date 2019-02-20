// RUN: %clang_cc1 -cfguard -emit-llvm %s -o - | FileCheck %s

void f() {}

// Check that the cfguardtable metadata flag gets set on the module.
// CHECK: !"cfguardtable", i32 1}
