// RUN: %clang_cc1 -cfguard-no-checks -emit-llvm %s -o - | FileCheck %s -check-prefix=CFGUARDNOCHECKS
// RUN: %clang_cc1 -cfguard -emit-llvm %s -o - | FileCheck %s -check-prefix=CFGUARD
// RUN: %clang_cc1 -ehcontguard -emit-llvm %s -o - | FileCheck %s -check-prefix=EHCONTGUARD

void f() {}

// Check that the cfguard metadata flag gets correctly set on the module.
// CFGUARDNOCHECKS: !"cfguard", i32 1}
// CFGUARD: !"cfguard", i32 2}
// EHCONTGUARD: !"ehcontguard", i32 1}
