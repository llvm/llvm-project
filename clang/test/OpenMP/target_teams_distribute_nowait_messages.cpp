// RUN: %clang_cc1 -verify -fopenmp -o - %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp-simd -o - %s -Wuninitialized

void foo() {
}

int main(int argc, char **argv) {
  int i, z;
#pragma omp target teams distribute nowait( //  expected-error {{expected expression}} //  expected-error {{expected ')'}} // expected-note {{to match this '('}}
#pragma omp target teams distribute nowait (argc)) // expected-warning {{extra tokens at the end of '#pragma omp target teams distribute' are ignored}} //expected-error {{region cannot be nested inside 'target teams distribute' region}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute nowait device (-10u)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute nowait (3.14) device (-10u)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute nowait (argc>> z)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute nowait (argv[1] = 2) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute nowait (argc > 0 ? argv[1] : argv[2])
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute nowait (S1) // expected-error {{use of undeclared identifier 'S1'}}
  for (i = 0; i < argc; ++i) foo();

  return 0;
}
