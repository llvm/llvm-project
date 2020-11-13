// RUN: %libomp-compile && env KMP_ABT_NUM_ESS=4 %libomp-run
// REQUIRES: abt
#include "omp_testsuite.h"
#include "bolt_scheduling_util.h"

int test_task_tied_thread_scheduling(int num_threads) {
  int vals[num_threads * num_threads];
  memset(vals, 0, sizeof(int) * num_threads * num_threads);
  omp_set_max_active_levels(2);

  timeout_barrier_t barrier;
  timeout_barrier_init(&barrier);

  #pragma omp parallel num_threads(num_threads)
  #pragma omp master
  {
    check_num_ess(4);
    int i;
    for (i = 0; i < num_threads; i++) {
      #pragma omp task firstprivate(i)
      {
        #pragma omp parallel num_threads(num_threads)
        {
          if (omp_get_thread_num() == 1) {
            // We should not block a master thread since it might need to create
            // other outer tasks.
            timeout_barrier_wait(&barrier, 4);
          }
          vals[i * num_threads + omp_get_thread_num()] += 1;
        }
      }
    }
  }

  #pragma omp parallel num_threads(num_threads)
  #pragma omp master
  {
    check_num_ess(4);
    int i;
    for (i = 0; i < num_threads; i++) {
      #pragma omp task firstprivate(i)
      {
        int j;
        #pragma omp parallel for num_threads(num_threads)
        for (j = 0; j < num_threads; j++) {
          if (omp_get_thread_num() == 1) {
            // We should not block a master thread since it might need to create
            // other outer tasks.
            timeout_barrier_wait(&barrier, 4);
          }
          vals[i * num_threads + j] += 2;
        }
      }
    }
  }

  int index;
  for (index = 0; index < num_threads * num_threads; index++) {
    if (vals[index] != 3) {
      printf("vals[%d] == %d\n", index, vals[index]);
      return 0;
    }
  }

  return 1;
}

int main() {
  int i, num_failed = 0;
  for (i = 1; i < 3; i++) {
    if (!test_task_tied_thread_scheduling(i * 4)) {
      num_failed++;
    }
  }
  return num_failed;
}
