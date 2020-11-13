// RUN: %libomp-compile && env KMP_ABT_NUM_ESS=4 %libomp-run
// REQUIRES: abt
#include "omp_testsuite.h"
#include "bolt_scheduling_util.h"

int test_thread_thread_scheduling(int num_threads) {
  int i, vals[num_threads * num_threads];
  memset(vals, 0, sizeof(int) * num_threads * num_threads);
  omp_set_max_active_levels(2);

  timeout_barrier_t barrier;
  timeout_barrier_init(&barrier);

  #pragma omp parallel num_threads(num_threads)
  {
    check_num_ess(4);
    int parent_tid = omp_get_thread_num();
    #pragma omp parallel num_threads(num_threads)
    {
      if (parent_tid == omp_get_thread_num()) {
        timeout_barrier_wait(&barrier, 4);
      }
      vals[parent_tid * num_threads + omp_get_thread_num()] += 1;
    }
  }

  #pragma omp parallel for num_threads(num_threads)
  for (i = 0; i < num_threads; i++) {
    check_num_ess(4);
    int j, parent_i = i;
    #pragma omp parallel for num_threads(num_threads)
    for (j = 0; j < num_threads; j++) {
      if (parent_i == j) {
        timeout_barrier_wait(&barrier, 4);
      }
      vals[parent_i * num_threads + j] += 2;
    }
  }

  for (i = 0; i < num_threads * num_threads; i++) {
    if (vals[i] != 3) {
      printf("vals[%d] == %d\n", i, vals[i]);
      return 0;
    }
  }

  return 1;
}

int main() {
  int i, num_failed = 0;
  for (i = 1; i < 3; i++) {
    if (!test_thread_thread_scheduling(i * 4)) {
      num_failed++;
    }
  }
  return num_failed;
}
