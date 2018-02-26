// RUN: %clang_cc1 -verify -fopenmp -o - %s

// RUN: %clang_cc1 -verify -fopenmp-simd -o - %s

void foo();

int main(int argc, char **argv) {
#pragma omp target teams default // expected-error {{expected '(' after 'default'}}
  foo();
#pragma omp target teams default ( // expected-error {{expected 'none' or 'shared' in OpenMP clause 'default'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
#pragma omp target teams default () // expected-error {{expected 'none' or 'shared' in OpenMP clause 'default'}}
  foo();
#pragma omp target teams default (none // expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
#pragma omp target teams default (shared), default(shared) // expected-error {{directive '#pragma omp target teams' cannot contain more than one 'default' clause}}
  foo();
#pragma omp target teams default (x) // expected-error {{expected 'none' or 'shared' in OpenMP clause 'default'}}
  foo();

#pragma omp target teams default(none)
  ++argc; // expected-error {{variable 'argc' must have explicitly specified data sharing attributes}}

#pragma omp target teams default(none)
#pragma omp parallel default(shared)
  ++argc;
  return 0;
}
