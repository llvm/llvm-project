// RUN: %clang_cc1 -import-call-optimization -emit-llvm %s -o - | FileCheck %s

void f(void) {}

// CHECK: !"import-call-optimization", i32 1}
