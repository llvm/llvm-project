// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify -fopenmp -ferror-limit 100 -o - %s

void foo() { }

int main(int argc, char **argv) {
  int a;
  #pragma omp target data // expected-error {{expected at least one 'map' or 'use_device_ptr' clause for '#pragma omp target data'}}
  {}
  L1:
    foo();
  #pragma omp target data map(a)
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
