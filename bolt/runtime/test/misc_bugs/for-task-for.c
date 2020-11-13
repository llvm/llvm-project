// RUN: %libomp-compile-and-run
#include <stdio.h>
#include <math.h>
#include "omp_testsuite.h"

#define NUM_OUTER_THREADS 16
#define NUM_INNER_THREADS 16
#define SMALL_LOOPCOUNT   64

/*! Utility function to spend some time in a loop */
static void do_some_work (void) {
  int i;
  double sum = 0;
  for(i = 0; i < 1000; i++) {
    sum += sqrt(i);
  }
}

int test_omp_parallel_for_task_for() {
  int vals[SMALL_LOOPCOUNT];
  int i;
  for (i = 0; i < SMALL_LOOPCOUNT; i++) {
    vals[i] = 0;
  }
  #pragma omp parallel firstprivate(vals) num_threads(NUM_OUTER_THREADS)
  #pragma omp master
  {
    for (i = 1; i <= SMALL_LOOPCOUNT; i++) {
      #pragma omp task firstprivate(i) firstprivate(vals)
      {
        int local_sum = 0;
        int j;
        #pragma omp parallel for reduction(+:local_sum) \
                num_threads(NUM_INNER_THREADS)
        for (j = 1; j <= SMALL_LOOPCOUNT; j++) {
          int k;
          do_some_work();
          for (k = 0; k < j % 4; k++) {
            #pragma omp taskyield
          }
          local_sum += j;
        }
        for (j = 0; j < i % 5; j++) {
          #pragma omp taskyield
        }
        vals[i] = local_sum;
      }
    }
  }
  int num_failed = 0;
  int known_sum = SMALL_LOOPCOUNT * (SMALL_LOOPCOUNT + 1) / 2;
  for (i = 0; i < SMALL_LOOPCOUNT; i++) {
    if (vals[i] != known_sum)
      num_failed++;
  }
  return num_failed ? 1 : 0;
}

int main() {
  int i;
  int num_failed = 0;

  for (i = 0; i < REPETITIONS; i++) {
    if (!test_omp_parallel_for_task_for()) {
      num_failed++;
    }
  }
  return num_failed;
}
