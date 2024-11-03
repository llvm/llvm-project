// RUN: %clang_cc1 -verify -fopenmp %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp-simd %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp-version=50 -DOMP50 -fopenmp %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp-version=50 -DOMP50 -fopenmp-simd %s -Wuninitialized

void foo();

namespace {
static int y = 0;
}
static int x = 0;

int main(int argc, char **argv) {
#pragma omp target teams distribute parallel for default // expected-error {{expected '(' after 'default'}}
  for (int i=0; i<200; i++) foo();
#pragma omp target teams distribute parallel for default( // expected-error {{expected 'none', 'shared', 'private' or 'firstprivate' in OpenMP clause 'default'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i=0; i<200; i++) foo();
#pragma omp target teams distribute parallel for default() // expected-error {{expected 'none', 'shared', 'private' or 'firstprivate' in OpenMP clause 'default'}}
  for (int i=0; i<200; i++) foo();
#pragma omp target teams distribute parallel for default (none // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i=0; i<200; i++) foo();
#pragma omp target teams distribute parallel for default (shared), default(shared) // expected-error {{directive '#pragma omp target teams distribute parallel for' cannot contain more than one 'default' clause}}
  for (int i=0; i<200; i++) foo();
#pragma omp target teams distribute parallel for default(x) // expected-error {{expected 'none', 'shared', 'private' or 'firstprivate' in OpenMP clause 'default'}}
  for (int i=0; i<200; i++) foo();

#pragma omp target teams distribute parallel for default(none) // expected-note {{explicit data sharing attribute requested here}}
  for (int i=0; i<200; i++) ++argc; // expected-error {{variable 'argc' must have explicitly specified data sharing attributes}}

#ifdef OMP50
#pragma omp target teams distribute parallel for default(firstprivate) // expected-error {{data-sharing attribute 'firstprivate' in 'default' clause requires OpenMP version 5.1 or above}}
  for (int i = 0; i < 200; i++) {
    ++x;
    ++y;
  }
#pragma omp target teams distribute parallel for default(private) // expected-error {{data-sharing attribute 'private' in 'default' clause requires OpenMP version 5.1 or above}}
  for (int i = 0; i < 200; i++) {
    ++x;
    ++y;
  }
#endif // OMP50

  return 0;
}
