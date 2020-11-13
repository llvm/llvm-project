// RUN: %libomp-compile-and-run
// REQUIRES: abt
#include "omp_testsuite.h"
#include <string.h>
#include <stdio.h>

int test_thread_threadid(int num_threads) {
  int i, vals[num_threads];
  memset(vals, 0, sizeof(int) * num_threads);

  #pragma omp parallel for num_threads(num_threads)
  for (i = 0; i < num_threads; i++) {
    int omp_thread_id = omp_get_thread_num();
    ABT_thread abt_thread;
    ABT_EXIT_IF_FAIL(ABT_thread_self(&abt_thread));

    // Context switching in OpenMP.
    #pragma omp taskyield

    int omp_thread_id2 = omp_get_thread_num();
    if (omp_thread_id == omp_thread_id2) {
      vals[i] += 1;
    }
    ABT_thread abt_thread2;
    ABT_EXIT_IF_FAIL(ABT_thread_self(&abt_thread2));
    ABT_bool abt_thread_equal;
    ABT_EXIT_IF_FAIL(ABT_thread_equal(abt_thread, abt_thread2,
                                      &abt_thread_equal));
    if (abt_thread_equal == ABT_TRUE) {
      vals[i] += 2;
    }


    // Context switching in Argobots.
    ABT_EXIT_IF_FAIL(ABT_thread_yield());

    int omp_thread_id3 = omp_get_thread_num();
    if (omp_thread_id2 == omp_thread_id3) {
      // Argobots context switch does not change the underlying thread.
      vals[i] += 4;
    }
  }

  for (i = 0; i < num_threads; i++) {
    if (vals[i] != 7) {
      printf("vals[%d] == %d\n", i, vals[i]);
      return 0;
    }
  }
  return 1;
}

int main() {
  int i, num_failed = 0;
  for (i = 0; i < REPETITIONS; i++) {
    if (!test_thread_threadid(i + 1)) {
      num_failed++;
    }
  }
  return num_failed;
}
