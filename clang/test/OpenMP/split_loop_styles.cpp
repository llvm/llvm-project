// Outer-declared iteration variable + split.
//
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -O0 -emit-llvm %s -o - | FileCheck %s

extern "C" void body(int);

// CHECK-LABEL: define {{.*}} @outer_iv(
// CHECK: .split.iv
extern "C" void outer_iv(int n) {
  int i;
#pragma omp split counts(3, omp_fill)
  for (i = 0; i < n; ++i)
    body(i);
}
