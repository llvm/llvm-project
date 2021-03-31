// test for r305179
// RUN: %clang_cc1 -emit-llvm -O2 -fno-experimental-new-pass-manager -mllvm -print-after-all %s -o %t 2>&1 | FileCheck %s
// CHECK: *** IR Dump After Function Integration/Inlining (inline) ***
void foo() {}
