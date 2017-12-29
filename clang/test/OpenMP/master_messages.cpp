// RUN: %clang_cc1 -verify -fopenmp %s

// RUN: %clang_cc1 -verify -fopenmp-simd %s

int foo();

int main() {
  #pragma omp master
  ;
  #pragma omp master nowait // expected-error {{unexpected OpenMP clause 'nowait' in directive '#pragma omp master'}}
  #pragma omp master unknown // expected-warning {{extra tokens at the end of '#pragma omp master' are ignored}}
  foo();
  {
    #pragma omp master
  } // expected-error {{expected statement}}
  #pragma omp for
  for (int i = 0; i < 10; ++i) {
    foo();
    #pragma omp master // expected-error {{region cannot be closely nested inside 'for' region}}
    foo();
  }
  #pragma omp sections
  {
    foo();
    #pragma omp master // expected-error {{region cannot be closely nested inside 'sections' region}}
    foo();
  }
  #pragma omp single
  for (int i = 0; i < 10; ++i) {
    foo();
    #pragma omp master // expected-error {{region cannot be closely nested inside 'single' region}}
    foo();
  }
  #pragma omp master
  for (int i = 0; i < 10; ++i) {
    foo();
    #pragma omp master
    foo();
  }
  #pragma omp for ordered
  for (int i = 0; i < 10; ++i)
  #pragma omp master // expected-error {{region cannot be closely nested inside 'for' region}}
  {
    foo();
  }

  return 0;
}

int foo() {
  L1:
    foo();
  #pragma omp master
  {
    foo();
    goto L1; // expected-error {{use of undeclared label 'L1'}}
  }
  goto L2; // expected-error {{use of undeclared label 'L2'}}
  #pragma omp master
  {
    L2:
    foo();
  }

  return 0;
}
