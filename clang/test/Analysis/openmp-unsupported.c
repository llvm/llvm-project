// RUN: %clang_cc1 -triple i386-apple-darwin10 -analyze -analyzer-checker=core.builtin -fopenmp -verify %s
// expected-no-diagnostics

void openmp_parallel_crash_test() {
#pragma omp parallel
  ;
}
