// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu \
// RUN:   -fopenmp -emit-llvm -o -  -x hip %s | FileCheck %s

#include "Inputs/cuda.h"

void foo(double) {}
__device__ void foo(int) {}

// Check foo resolves to the host function.
// CHECK-LABEL: define {{.*}}@_Z5test1v
// CHECK: call void @_Z3food(double noundef 1.000000e+00)
void test1() {
  #pragma omp parallel
  for (int i = 0; i < 100; i++)
    foo(1);
}

// Check foo resolves to the host function.
// CHECK-LABEL: define {{.*}}@_Z5test2v
// CHECK: call void @_Z3food(double noundef 1.000000e+00)
void test2() {
  auto Lambda = []() {
    #pragma omp parallel
    for (int i = 0; i < 100; i++)
      foo(1);
  };
  Lambda();
}
