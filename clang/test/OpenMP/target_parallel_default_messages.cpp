// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 -o - %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 -o - %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp-version=51 -DOMP51 -fopenmp -ferror-limit 100 -o - %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp-version=51 -DOMP51 -fopenmp-simd -ferror-limit 100 -o - %s -Wuninitialized

void foo();

namespace {
static int y = 0;
}
static int x = 0;

int main(int argc, char **argv) {
  #pragma omp target parallel default // expected-error {{expected '(' after 'default'}}
  foo();
#pragma omp target parallel default( // expected-error {{expected 'none', 'shared' or 'firstprivate' in OpenMP clause 'default'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
#pragma omp target parallel default() // expected-error {{expected 'none', 'shared' or 'firstprivate' in OpenMP clause 'default'}}
  foo();
  #pragma omp target parallel default (none // expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target parallel default (shared), default(shared) // expected-error {{directive '#pragma omp target parallel' cannot contain more than one 'default' clause}}
  foo();
#pragma omp target parallel default(x) // expected-error {{expected 'none', 'shared' or 'firstprivate' in OpenMP clause 'default'}}
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

#ifndef OMP51
#pragma omp target parallel default(firstprivate) // expected-error {{data-sharing attribute 'firstprivate' in 'default' clause requires OpenMP version 5.1 or above}}
  {
    ++x;
    ++y;
  }
#endif

  return 0;
}
