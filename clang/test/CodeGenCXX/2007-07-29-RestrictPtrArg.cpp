// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s

void foo(int * __restrict myptr1, int * myptr2) {
  // CHECK: noalias
  myptr1[0] = 0;
  myptr2[0] = 0;
}
