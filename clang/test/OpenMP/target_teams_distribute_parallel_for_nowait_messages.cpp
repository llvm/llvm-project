// RUN: %clang_cc1 -fsyntax-only -verify -fopenmp %s -Wuninitialized

// RUN: %clang_cc1 -fsyntax-only -verify -fopenmp-simd %s -Wuninitialized

void foo() {
}

int main(int argc, char **argv) {
  int i, z;
#pragma omp target teams distribute parallel for nowait( //  expected-error {{expected expression}} //  expected-error {{expected ')'}} // expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for nowait (argc)) // expected-warning {{extra tokens at the end of '#pragma omp target teams distribute parallel for' are ignored}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for nowait device (-10u)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for nowait (3.14) device (-10u)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for nowait (argc>> z)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for nowait (argv[1] = 2) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for nowait (argc > 0 ? argv[1] : argv[2])
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for nowait (S1) // expected-error {{use of undeclared identifier 'S1'}}
  for (i = 0; i < argc; ++i) foo();

  return 0;
}
