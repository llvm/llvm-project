// test for r305179
// RUN: %clang_cc1 -emit-llvm -O -mllvm -print-after-all %s -o %t 2>&1 | FileCheck %s
// CHECK: *** IR Dump After Function Integration/Inlining ***
void foo() {}
