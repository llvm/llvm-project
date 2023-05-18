// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=50 -o - %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=50 -o - %s -Wuninitialized

// RUN: %clang_cc1 -verify -DOMP51 -fopenmp -o - %s -Wuninitialized

// RUN: %clang_cc1 -verify -DOMP51 -fopenmp-simd -o - %s -Wuninitialized

void foo();

namespace {
static int y = 0;
}
static int x = 0;

int main(int argc, char **argv) {
  int i;
#pragma omp parallel for simd default // expected-error {{expected '(' after 'default'}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp parallel for simd default( // expected-error {{expected 'none', 'shared', 'private' or 'firstprivate' in OpenMP clause 'default'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp parallel for simd default() // expected-error {{expected 'none', 'shared', 'private' or 'firstprivate' in OpenMP clause 'default'}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp parallel for simd default(none // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-note {{explicit data sharing attribute requested here}}
  for (i = 0; i < argc; ++i) // expected-error {{variable 'argc' must have explicitly specified data sharing attributes}}
    foo();
#pragma omp parallel for simd default(shared), default(shared) // expected-error {{directive '#pragma omp parallel for simd' cannot contain more than one 'default' clause}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp parallel for simd default(x) // expected-error {{expected 'none', 'shared', 'private' or 'firstprivate' in OpenMP clause 'default'}}
  for (i = 0; i < argc; ++i)
    foo();

#pragma omp parallel for simd default(none) // expected-note {{explicit data sharing attribute requested here}}
  for (i = 0; i < argc; ++i)  // expected-error {{variable 'argc' must have explicitly specified data sharing attributes}}
    foo();

#pragma omp parallel default(none) // expected-note 2 {{explicit data sharing attribute requested here}}
#pragma omp parallel for simd default(shared)
  for (i = 0; i < argc; ++i) // expected-error {{variable 'argc' must have explicitly specified data sharing attributes}} expected-error {{variable 'i' must have explicitly specified data sharing attributes}}
    foo();

#ifdef OMP51
#pragma omp parallel for default(firstprivate) // expected-note 2 {{explicit data sharing attribute requested here}}
  for (i = 0; i < argc; ++i) {
    x++; // expected-error {{variable 'x' must have explicitly specified data sharing attributes}}
    y++; // expected-error {{variable 'y' must have explicitly specified data sharing attributes}}
  }
#pragma omp parallel for default(private) // expected-note 2 {{explicit data sharing attribute requested here}}
  for (i = 0; i < argc; ++i) {
    x++; // expected-error {{variable 'x' must have explicitly specified data sharing attributes}}
    y++; // expected-error {{variable 'y' must have explicitly specified data sharing attributes}}
  }
#endif

  return 0;
}
