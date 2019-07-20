// RUN: %clang_cc1 -triple i386-apple-darwin10 -analyze -analyzer-checker=core.builtin -fopenmp -verify %s
// expected-no-diagnostics

void openmp_parallel_crash_test() {
#pragma omp parallel
  ;
#pragma omp parallel for
  for (int i = 0; i < 8; ++i)
    for (int j = 0, k = 0; j < 8; ++j)
      ;
}
