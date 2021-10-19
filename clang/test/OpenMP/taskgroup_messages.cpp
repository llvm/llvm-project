// RUN: %clang_cc1 -verify -fopenmp %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp-simd %s -Wuninitialized

void xxx(int argc) {
  int x; // expected-note {{initialize the variable 'x' to silence this warning}}
#pragma omp taskgroup
  argc = x; // expected-warning {{variable 'x' is uninitialized when used here}}
}

int foo();

int main() {
  #pragma omp taskgroup
  ;
  #pragma omp taskgroup unknown // expected-warning {{extra tokens at the end of '#pragma omp taskgroup' are ignored}}
  foo();
  {
    #pragma omp taskgroup
  } // expected-error {{expected statement}}
  #pragma omp taskgroup
  #pragma omp taskgroup
  for (int i = 0; i < 10; ++i) {
    foo();
    #pragma omp parallel
    #pragma omp for
    for (int j = 0; j < 10; j++) {
      foo();
      #pragma omp taskgroup
      foo();
    }
  }
  #pragma omp taskgroup
  #pragma omp taskgroup
  for (int i = 0; i < 10; ++i) {
    foo();
    #pragma omp parallel
    #pragma omp for
    for (int j = 0; j < 10; j++) {
      #pragma omp taskgroup
      foo();
    }
  }
  #pragma omp taskgroup
  #pragma omp taskgroup
  for (int i = 0; i < 10; ++i) {
    foo();
    #pragma omp parallel
    #pragma omp for
    for (int j = 0; j < 10; j++) {
      #pragma omp taskgroup
      foo();
    }
  }

  return 0;
}

int foo() {
  L1:
    foo();
  #pragma omp taskgroup
  {
    foo();
    goto L1; // expected-error {{use of undeclared label 'L1'}}
  }
  goto L2; // expected-error {{use of undeclared label 'L2'}}
  #pragma omp taskgroup
  {
    L2:
    foo();
  }

#pragma omp taskgroup initi // expected-warning {{extra tokens at the end of '#pragma omp taskgroup' are ignored}}
  ;
  return 0;
}
