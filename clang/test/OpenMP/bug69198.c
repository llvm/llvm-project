// RUN: %clang_cc1 -verify -fopenmp -x c -triple x86_64-apple-darwin10 %s
// expected-no-diagnostics

int c[-1];

void foo (){
  #pragma omp task depend(inout: c[:][:])
}
