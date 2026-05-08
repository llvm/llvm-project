// Driver forwards `-fopenmp-version=60` with split source (`###` only — no link).
// REQUIRES: x86-registered-target
//
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp -fopenmp-version=60 -c %s -o %t.o 2>&1 | FileCheck %s --check-prefix=INVOC

void f(int n) {
#pragma omp split counts(2, omp_fill)
  for (int i = 0; i < n; ++i) {
  }
}

// INVOC: -fopenmp-version=60
