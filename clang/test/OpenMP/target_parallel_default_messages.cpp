// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 -o - %s

// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 -o - %s

void foo();

int main(int argc, char **argv) {
  #pragma omp target parallel default // expected-error {{expected '(' after 'default'}}
  foo();
  #pragma omp target parallel default ( // expected-error {{expected 'none' or 'shared' in OpenMP clause 'default'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target parallel default () // expected-error {{expected 'none' or 'shared' in OpenMP clause 'default'}}
  foo();
  #pragma omp target parallel default (none // expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target parallel default (shared), default(shared) // expected-error {{directive '#pragma omp target parallel' cannot contain more than one 'default' clause}}
  foo();
  #pragma omp target parallel default (x) // expected-error {{expected 'none' or 'shared' in OpenMP clause 'default'}}
  foo();

  #pragma omp target parallel default(none) // expected-note {{explicit data sharing attribute requested here}}
  ++argc; // expected-error {{variable 'argc' must have explicitly specified data sharing attributes}}

  #pragma omp target parallel default(none)
  foo();
  #pragma omp target parallel default(shared)
  ++argc;
  #pragma omp target parallel default(none) // expected-note {{explicit data sharing attribute requested here}}
  #pragma omp parallel default(shared)
  ++argc; // expected-error {{variable 'argc' must have explicitly specified data sharing attributes}}
  return 0;
}
