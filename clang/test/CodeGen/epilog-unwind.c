// RUN: %clang_cc1 -winx64-eh-unwindv2 -emit-llvm %s -o - | FileCheck %s

void f(void) {}

// CHECK: !"winx64-eh-unwindv2", i32 1}
