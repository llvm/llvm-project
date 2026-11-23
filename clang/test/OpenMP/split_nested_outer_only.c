// Split attaches to the outer canonical `for`; inner loop stays unsplit.
//
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -O0 -emit-llvm %s -o - | FileCheck %s

// Exactly one split IV — the outer loop; inner `for` uses plain `i`/`j` control flow.
// CHECK-COUNT-1: .split.iv
void f(void) {
#pragma omp split counts(omp_fill)
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) {
    }
}
