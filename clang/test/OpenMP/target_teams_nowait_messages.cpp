// RUN: %clang_cc1 -verify -fopenmp -o - %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp-simd -o - %s -Wuninitialized

void foo() {
}

int main(int argc, char **argv) {
  int z;
#pragma omp target teams nowait( // expected-error {{expected expression}} // expected-error {{expected ')'}} //expected-note {{to match this '('}}
  foo();
#pragma omp target teams nowait device (-10u)
  foo();
#pragma omp target teams nowait (3.14) device (-10u)
  foo();
#pragma omp target teams nowait (argc>> z)
  foo();
#pragma omp target teams nowait (argv[1] = 2) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
#pragma omp target teams nowait (argc > 0 ? argv[1] : argv[2])
  foo();
#pragma omp target teams nowait (S1) // expected-error {{use of undeclared identifier 'S1'}}
  foo();

  return 0;
}
