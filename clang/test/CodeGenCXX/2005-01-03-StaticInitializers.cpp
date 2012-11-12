// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s
// REQUIRES: LP64

struct S {
  int  A[2];
};

// CHECK-NOT: llvm.global_ctor
int XX = (int)(long)&(((struct S*)0)->A[1]);
