// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify -fopenmp -ferror-limit 100 -o - %s -Wuninitialized

// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify -fopenmp-simd -ferror-limit 100 -o - %s -Wuninitialized

void foo() { }

void xxx(int argc) {
  int map; // expected-note {{initialize the variable 'map' to silence this warning}}
#pragma omp target data map(map) // expected-warning {{variable 'map' is uninitialized when used here}}
  for (int i = 0; i < 10; ++i)
    ;
}

int main(int argc, char **argv) {
  int a;
  #pragma omp target data // expected-error {{expected at least one 'map' or 'use_device_ptr' clause for '#pragma omp target data'}}
  {}
  L1:
    foo();
  #pragma omp target data map(a) allocate(a) // expected-error {{unexpected OpenMP clause 'allocate' in directive '#pragma omp target data'}}
  {
    foo();
    goto L1; // expected-error {{use of undeclared label 'L1'}}
  }
  goto L2; // expected-error {{use of undeclared label 'L2'}}
  #pragma omp target data map(a)
  L2:
  foo();

  #pragma omp target data map(a)(i) // expected-warning {{extra tokens at the end of '#pragma omp target data' are ignored}}
  {
    foo();
  }
  #pragma omp target unknown // expected-warning {{extra tokens at the end of '#pragma omp target' are ignored}}
  {
    foo();
  }
  return 0;
}
