// RUN: %clang_cc1 -fopenmp -x c++ %s -verify -debug-info-kind=limited -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

void f(int m) {
  int i;
  int cen[m];
#pragma omp parallel for
  for (i = 0; i < m; ++i) {
    cen[i] = i;
  }
}

// CHECK: !DILocalVariable(name: "cen", arg: 6
