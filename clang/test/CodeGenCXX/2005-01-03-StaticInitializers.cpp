// RUN: %clang_cc1 -triple=x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple=i386-linux-gnu -emit-llvm %s -o - | FileCheck %s

struct S {
  int  A[2];
};

// CHECK: @XX = global i32 4, align 4
int XX = (int)(long)&(((struct S*)0)->A[1]);
