// RUN: %clang_cc1 -verify -fopenmp -std=c++11 -fsyntax-only %s

// expected-no-diagnostics

enum omp_interop_t : unsigned long {};

template <typename T> void foo() {
  T t;
#pragma omp interop init(target : t)
}

void bar(int *y) {
  foo<omp_interop_t>();
  --y;
}
