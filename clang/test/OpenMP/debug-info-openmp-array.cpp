// RUN: %clang_cc1 -fopenmp -x c++ %s -verify -debug-info-kind=limited -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -fopenmp-simd -x c++ %s -verify -debug-info-kind=limited -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics

void f(int m) {
  int i;
  int cen[m];
#pragma omp parallel for
  for (i = 0; i < m; ++i) {
    cen[i] = i;
  }
}

// CHECK: !DILocalVariable(name: "cen", arg: 5
