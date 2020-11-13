// RUN: %libomp-compile-and-run
// REQUIRES: abt && !clang

// Clang 10.0 seems ignoring the taskloop's "untied" attribute.
// We mark taskloop + untied with Clang as unsupported so far.
#include "omp_testsuite.h"
#include <string.h>
#include <stdio.h>

int test_taskloop_untied_threadid(int num_threads) {
  int vals[NUM_TASKS];
  memset(vals, 0, sizeof(vals));

  #pragma omp parallel num_threads(num_threads)
  {
    #pragma omp master
    {
      int i;
      #pragma omp taskloop grainsize(1) untied
      for (i = 0; i < NUM_TASKS; i++) {
        {
          ABT_thread abt_thread;
          ABT_EXIT_IF_FAIL(ABT_thread_self(&abt_thread));

          // Context switching in OpenMP.
          #pragma omp taskyield

          int omp_thread_id2 = omp_get_thread_num();
          ABT_thread abt_thread2;
          ABT_EXIT_IF_FAIL(ABT_thread_self(&abt_thread2));
          ABT_bool abt_thread_equal;
          ABT_EXIT_IF_FAIL(ABT_thread_equal(abt_thread, abt_thread2,
                                            &abt_thread_equal));
          if (abt_thread_equal == ABT_TRUE) {
            vals[i] += 1;
          }

          // Context switching in Argobots.
          ABT_EXIT_IF_FAIL(ABT_thread_yield());

          int omp_thread_id3 = omp_get_thread_num();
          if (omp_thread_id2 == omp_thread_id3) {
            // Argobots context switch does not change the thread-task mapping.
            vals[i] += 2;
          }
        }
      }
    }
  }

  int index;
  for (index = 0; index < NUM_TASKS; index++) {
    if (vals[index] != 3) {
      printf("vals[%d] == %d\n", index, vals[index]);
      return 0;
    }
  }
  return 1;
}

int main() {
  int i;
  int num_failed = 0;
  for (i = 0; i < REPETITIONS; i++) {
    if (!test_taskloop_untied_threadid(i + 1)) {
      num_failed++;
    }
  }
  return num_failed;
}
