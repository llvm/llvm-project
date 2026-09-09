// RUN: %clang_cc1 %s -verify -fopenmp

struct Type {
  void foo() {
#pragma omp parallel private(bar })
    // expected-error@-1{{use of undeclared identifier 'bar'}}
    // expected-error@+1{{expected statement}}
  }
};
