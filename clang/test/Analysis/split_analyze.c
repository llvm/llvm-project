// Static analyzer invocation on split loop.
// RUN: %clang_analyze_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -fopenmp-version=60 -analyzer-checker=core.builtin -verify %s
// expected-no-diagnostics

void g(int);

void f(int n) {
#pragma omp split counts(2, omp_fill)
  for (int i = 0; i < n; ++i)
    g(i);
}
