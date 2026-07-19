// `#pragma omp split` / `counts` require OpenMP 6.x in this implementation.
//
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -std=c++17 -fopenmp -fopenmp-version=60 -fsyntax-only -DONLY_OK -verify %s
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -std=c++17 -fopenmp -fopenmp-version=52 -fsyntax-only -DONLY_BAD52 -verify=expected52 %s

#if defined(ONLY_OK)
void ok60(void) {
#pragma omp split counts(omp_fill)
  for (int i = 0; i < 10; ++i) {
  }
}
// expected-no-diagnostics
#endif

#if defined(ONLY_BAD52)
// expected52-error@+2 {{unexpected OpenMP clause 'counts' in directive '#pragma omp split'}}
void bad52(void) {
#pragma omp split counts(omp_fill)
  for (int i = 0; i < 10; ++i) {
  }
}
#endif
