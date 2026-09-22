// RUN: %clang_omp_offload_csan -fno-exceptions %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

#include "race.h"

int main() {
  int X = 0;
#pragma omp target teams num_teams(1) thread_limit(64) map(tofrom : X)
#pragma omp parallel num_threads(64)
  {
    volatile int *P = (volatile int *)&X;
    RACE_UNTIL_FOUND(I)
    *P = I;
  }
  return 0;
}

// CHECK: WARNING: ConcurrencySanitizer: {{.*}}race
// CHECK: #0 {{.*}}openmp-race.cpp:{{[0-9]+}}
