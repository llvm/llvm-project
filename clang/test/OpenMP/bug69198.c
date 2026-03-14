// RUN: %clang_cc1 -verify -fsyntax-only -fopenmp -x c %s

int c[-1]; // expected-error {{'c' declared as an array with a negative size}}

void foo (){
  #pragma omp task depend(inout: c[:][:])
  {
    c[0] = 1;
  }
}
