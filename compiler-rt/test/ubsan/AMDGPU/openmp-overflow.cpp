// REQUIRES: ubsan-openmp-offload
// RUN: %clang_ubsan_omp %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

#include <omp.h>

int main() {
  int X = 0;
#pragma omp target map(from : X)
  {
    int A = 0x7fffffff;
    X = A + 1;
  }
  return 0;
}

// CHECK: runtime error: signed integer overflow
// CHECK: SUMMARY: UndefinedBehaviorSanitizer:
