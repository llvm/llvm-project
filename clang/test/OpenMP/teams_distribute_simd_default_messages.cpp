// RUN: %clang_cc1 -verify -fopenmp %s -Wuninitialized -fopenmp-version=51

// RUN: %clang_cc1 -verify -fopenmp-simd %s -Wuninitialized -fopenmp-version=51

void foo();

namespace {
static int y = 0;
}
static int x = 0;

int main(int argc, char **argv) {
  #pragma omp target
  #pragma omp teams distribute simd default // expected-error {{expected '(' after 'default'}}
  for (int i=0; i<200; i++) foo();
  #pragma omp target
#pragma omp teams distribute simd default( // expected-error {{expected 'none', 'shared', 'private' or 'firstprivate' in OpenMP clause 'default'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i=0; i<200; i++) foo();
  #pragma omp target
#pragma omp teams distribute simd default() // expected-error {{expected 'none', 'shared', 'private' or 'firstprivate' in OpenMP clause 'default'}}
  for (int i=0; i<200; i++) foo();
  #pragma omp target
  #pragma omp teams distribute simd default (none // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i=0; i<200; i++) foo();
  #pragma omp target
  #pragma omp teams distribute simd default (shared), default(shared) // expected-error {{directive '#pragma omp teams distribute simd' cannot contain more than one 'default' clause}}
  for (int i=0; i<200; i++) foo();
  #pragma omp target
#pragma omp teams distribute simd default(x) // expected-error {{expected 'none', 'shared', 'private' or 'firstprivate' in OpenMP clause 'default'}}
  for (int i=0; i<200; i++) foo();

  #pragma omp target
  #pragma omp teams distribute simd default(none) // expected-note {{explicit data sharing attribute requested here}}
  for (int i=0; i<200; i++) ++argc; // expected-error {{variable 'argc' must have explicitly specified data sharing attributes}}

#pragma omp target
#pragma omp teams distribute simd default(firstprivate) // expected-note {{explicit data sharing attribute requested here}}
  for (int i = 0; i < 200; i++)
    ++x; // expected-error {{variable 'x' must have explicitly specified data sharing attributes}}

#pragma omp target
#pragma omp teams distribute simd default(firstprivate) // expected-note {{explicit data sharing attribute requested here}}
  for (int i = 0; i < 200; i++)
    ++y; // expected-error {{variable 'y' must have explicitly specified data sharing attributes}}

#pragma omp target
#pragma omp teams distribute simd default(private) // expected-note {{explicit data sharing attribute requested here}}
  for (int i = 0; i < 200; i++)
    ++x; // expected-error {{variable 'x' must have explicitly specified data sharing attributes}}

#pragma omp target
#pragma omp teams distribute simd default(private) // expected-note {{explicit data sharing attribute requested here}}
  for (int i = 0; i < 200; i++)
    ++y; // expected-error {{variable 'y' must have explicitly specified data sharing attributes}}

  return 0;
}
