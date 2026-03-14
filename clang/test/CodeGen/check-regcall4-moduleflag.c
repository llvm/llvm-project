// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s -check-prefix=NO-REGCALL4
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -regcall4 -emit-llvm %s -o - | FileCheck %s -check-prefix=REGCALL4

void f(void) {}

// REGCALL4: !"RegCallv4", i32 1}
// NO-REGCALL4-NOT: "RegCallv4"
