// RUN: %clang_cc1 -verify -fopenmp -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++11 -include-pch %t -verify %s -ast-print | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

int main() {
  int x = 0, y = 0;

#pragma omp taskgraph
// CHECK: #pragma omp taskgraph
  {
#pragma omp task depend(in: x) depend(out: y)
// CHECK: #pragma omp task depend(in : x) depend(out : y)
    {
      y = x;
    }
#pragma omp task depend(inout: x, y)
// CHECK: #pragma omp task depend(inout : x,y)
    {
      x++;
      y++;
    }
  }

  return 0;
}

#endif
