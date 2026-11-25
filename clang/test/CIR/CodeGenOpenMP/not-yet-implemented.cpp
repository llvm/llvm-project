// RUN: %clang_cc1 -fopenmp -triple x86_64-linux-gnu -fclangir %s -verify -emit-cir -o -

void range_for() {
  int a[10];
#pragma omp parallel for
  // expected-error@+1{{ClangIR code gen Not Yet Implemented: range-based for loop in an OpenMP loop directive}}
  for (int &x : a)
    x = 1;
}
