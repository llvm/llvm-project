// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 -o - %s

// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 -o - %s

void foo();

int main(int argc, char **argv) {
  #pragma omp parallel default // expected-error {{expected '(' after 'default'}}
  #pragma omp parallel default ( // expected-error {{expected 'none' or 'shared' in OpenMP clause 'default'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp parallel default () // expected-error {{expected 'none' or 'shared' in OpenMP clause 'default'}}
  #pragma omp parallel default (none // expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp parallel default (shared), default(shared) // expected-error {{directive '#pragma omp parallel' cannot contain more than one 'default' clause}}
  #pragma omp parallel default (x) // expected-error {{expected 'none' or 'shared' in OpenMP clause 'default'}}
  foo();

  #pragma omp parallel default(none)
  ++argc; // expected-error {{variable 'argc' must have explicitly specified data sharing attributes}}

  #pragma omp parallel default(none)
  #pragma omp parallel default(shared)
  ++argc; // expected-error {{variable 'argc' must have explicitly specified data sharing attributes}}
  return 0;
}
