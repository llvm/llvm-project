// RUN: %clang_cc1 -verify -fopenmp -x c -emit-llvm %s -triple x86_64-unknown-linux -o - -femit-all-decls -disable-llvm-passes | FileCheck %s
// RUN: %clang_cc1 -verify -x c -emit-llvm %s -triple x86_64-unknown-linux -o - -femit-all-decls -disable-llvm-passes | FileCheck %s
// expected-no-diagnostics

// CHECK: !{{[0-9]+}} = !{!"llvm.loop.vectorize.width", i32 1}
void sub(double *restrict a, double *restrict b, int n) {
  int i;

#pragma omp parallel for
#pragma clang loop vectorize(disable)
  for (i = 0; i < n; i++) {
    a[i] = a[i] + b[i];
  }
}
