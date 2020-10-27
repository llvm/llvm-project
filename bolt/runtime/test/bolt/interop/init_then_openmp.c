// RUN: %libomp-compile-and-run
// REQUIRES: abt
#include "omp_testsuite.h"
#include <stdio.h>

int test_init_then_openmp(int num_init) {
  int i;
  int val = 0;

  for (i = 0; i < num_init; i++) {
    ABT_EXIT_IF_FAIL(ABT_init(0, 0));
  }

  #pragma omp parallel num_threads(NUM_TASKS)
  {
    #pragma omp master
    { val = 1; }
  }

  for (i = 0; i < num_init; i++) {
    ABT_EXIT_IF_FAIL(ABT_finalize());
  }

  return val;
}

int main() {
  int i;
  int num_failed = 0;
  for (i = 0; i < REPETITIONS; i++) {
    // Note that Argobots will be initialized once BOLT is instantiated.
    if (!test_init_then_openmp(i + 1)) {
      num_failed++;
    }
  }
  return num_failed;
}
