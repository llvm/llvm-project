// RUN: %clang_cc1 -verify -fopenmp %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp-simd %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp %s -Wuninitialized -fopenmp-version=51 -DOMP51

// RUN: %clang_cc1 -verify -fopenmp-simd %s -Wuninitialized -fopenmp-version=51 -DOMP51

void foo();

namespace {
static int y = 0;
}
static int x = 0;

int main(int argc, char **argv) {
  #pragma omp target
  #pragma omp teams distribute parallel for simd default // expected-error {{expected '(' after 'default'}}
  for (int i=0; i<200; i++) foo();
  #pragma omp target
#pragma omp teams distribute parallel for simd default( // expected-error {{expected 'none', 'shared', 'private' or 'firstprivate' in OpenMP clause 'default'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i=0; i<200; i++) foo();
  #pragma omp target
#pragma omp teams distribute parallel for simd default() // expected-error {{expected 'none', 'shared', 'private' or 'firstprivate' in OpenMP clause 'default'}}
  for (int i=0; i<200; i++) foo();
  #pragma omp target
  #pragma omp teams distribute parallel for simd default (none // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i=0; i<200; i++) foo();
  #pragma omp target
  #pragma omp teams distribute parallel for simd default (shared), default(shared) // expected-error {{directive '#pragma omp teams distribute parallel for simd' cannot contain more than one 'default' clause}}
  for (int i=0; i<200; i++) foo();
  #pragma omp target
#pragma omp teams distribute parallel for simd default(x) // expected-error {{expected 'none', 'shared', 'private' or 'firstprivate' in OpenMP clause 'default'}}
  for (int i=0; i<200; i++) foo();

  #pragma omp target
  #pragma omp teams distribute parallel for simd default(none) // expected-note {{explicit data sharing attribute requested here}}
  for (int i=0; i<200; i++) ++argc; // expected-error {{variable 'argc' must have explicitly specified data sharing attributes}}

#ifdef OpenMP51
#pragma omp teams distribute parallel for default(firstprivate) // expected-note 2 {{explicit data sharing attribute requested here}}
  for (int i = 0; i < 200; i++) {
    ++x; // expected-error {{variable 'x' must have explicitly specified data sharing attributes}}
    ++y; // expected-error {{variable 'y' must have explicitly specified data sharing attributes}}
  }
#pragma omp teams distribute parallel for default(private) // expected-note 2 {{explicit data sharing attribute requested here}}
  for (int i = 0; i < 200; i++) {
    ++x; // expected-error {{variable 'x' must have explicitly specified data sharing attributes}}
    ++y; // expected-error {{variable 'y' must have explicitly specified data sharing attributes}}
  }
#endif

  return 0;
}
