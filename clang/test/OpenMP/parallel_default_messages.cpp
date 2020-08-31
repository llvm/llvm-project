// RUN: %clang_cc1 -verify=expected,ge40 -fopenmp-version=45 -fopenmp -ferror-limit 100 -o - %s -Wuninitialized
// RUN: %clang_cc1 -verify=expected,ge40 -fopenmp-simd -ferror-limit 100 -o - %s -Wuninitialized
// RUN: %clang_cc1 -verify=expected,ge40 -fopenmp -ferror-limit 100 -o - %s -Wuninitialized
// RUN: %clang_cc1 -verify=expected,ge40 -fopenmp-version=40 -fopenmp -ferror-limit 100 -o - %s -Wuninitialized
// RUN: %clang_cc1 -verify -fopenmp-version=31 -fopenmp -ferror-limit 100 -o - %s -Wuninitialized
// RUN: %clang_cc1 -verify -fopenmp-version=30 -fopenmp -ferror-limit 100 -o - %s -Wuninitialized
// RUN: %clang_cc1 -verify=expected,ge40 -fopenmp-version=51 -fopenmp -DOMP51 -ferror-limit 100 -o - %s -Wuninitialized
// RUN: %clang_cc1 -verify=expected,ge40 -fopenmp-version=51 -fopenmp-simd -DOMP51 -ferror-limit 100 -o - %s -Wuninitialized

void foo();

namespace {
static int y = 0;
}
static int x = 0;

int main(int argc, char **argv) {
  const int c = 0;

  #pragma omp parallel default // expected-error {{expected '(' after 'default'}}
#pragma omp parallel default(  // expected-error {{expected 'none', 'shared' or 'firstprivate' in OpenMP clause 'default'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp parallel default() // expected-error {{expected 'none', 'shared' or 'firstprivate' in OpenMP clause 'default'}}
#pragma omp parallel default(none                     // expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp parallel default(shared), default(shared) // expected-error {{directive '#pragma omp parallel' cannot contain more than one 'default' clause}}
#pragma omp parallel default(x)                       // expected-error {{expected 'none', 'shared' or 'firstprivate' in OpenMP clause 'default'}}
  foo();

  #pragma omp parallel default(none) // expected-note {{explicit data sharing attribute requested here}}
  ++argc; // expected-error {{variable 'argc' must have explicitly specified data sharing attributes}}

  #pragma omp parallel default(none) // expected-note {{explicit data sharing attribute requested here}}
  #pragma omp parallel default(shared)
  ++argc; // expected-error {{variable 'argc' must have explicitly specified data sharing attributes}}

  #pragma omp parallel default(none) // ge40-note {{explicit data sharing attribute requested here}}
  (void)c; // ge40-error {{variable 'c' must have explicitly specified data sharing attributes}}

#ifdef OMP51
#pragma omp parallel default(firstprivate) // expected-note 2 {{explicit data sharing attribute requested here}}
  {
    ++x; // expected-error {{variable 'x' must have explicitly specified data sharing attributes}}
    ++y; // expected-error {{variable 'y' must have explicitly specified data sharing attributes}}
  }
#endif

  return 0;
}
