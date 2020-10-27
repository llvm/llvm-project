// RUN: %libomp-compile && env KMP_ABT_NUM_ESS=4 %libomp-run
// REQUIRES: abt
#include "omp_testsuite.h"
#include "bolt_scheduling_util.h"

int test_for_nowait_scheduling() {
  int i, vals[4];
  memset(vals, 0, sizeof(int) * 4);

  timeout_barrier_t barrier;
  timeout_barrier_init(&barrier);

  #pragma omp parallel num_threads(4)
  {
    check_num_ess(4);
    int tid = omp_get_thread_num();
    #pragma omp for nowait
    for (i = 0; i < 4; i++) {
      if (tid < 2) {
        timeout_barrier_wait(&barrier, 4);
      }
    }
    if (tid >= 2) {
      // The following barrier must be synchronized with the "for" above.
      timeout_barrier_wait(&barrier, 4);
    }
    vals[omp_get_thread_num()] = 1;
  }

  for (i = 0; i < 4; i++) {
    if (vals[i] != 1) {
      printf("vals[%d] == %d\n", i, vals[i]);
      return 0;
    }
  }
  return 1;
}

int main() {
  int i, num_failed = 0;
  for (i = 1; i < REPETITIONS; i++) {
    if (!test_for_nowait_scheduling(i)) {
      num_failed++;
    }
  }
  return num_failed;
}
