// RUN: %libomp-compile && env KMP_ABT_NUM_ESS=4 %libomp-run
// REQUIRES: abt && !clang

// Clang 10.0 seems ignoring the taskloop's "untied" attribute.
// We mark taskloop + untied with Clang as unsupported so far.
#include "omp_testsuite.h"
#include "bolt_scheduling_util.h"

int test_taskloop_nogroup_untied_scheduling() {
  int i, vals[6];
  memset(vals, 0, sizeof(int) * 6);

  timeout_barrier_t barrier;
  timeout_barrier_init(&barrier);

  #pragma omp parallel num_threads(4)
  {
    // 6 barrier_waits in tasks and 2 barrier_waits in threads
    #pragma omp master
    {
      check_num_ess(4);
      #pragma omp taskloop grainsize(1) nogroup untied
      for (i = 0; i < 6; i++) {
        timeout_barrier_wait(&barrier, 4);
        vals[i] = 1;
      }
    }
    if (omp_get_thread_num() < 2) {
      // master does not wait the completion of taskloop.
      timeout_barrier_wait(&barrier, 4);
    }
  }

  #pragma omp parallel num_threads(4)
  {
    // 6 barrier_waits in tasks and 2 barrier_waits in threads
    #pragma omp master
    {
      check_num_ess(4);
      #pragma omp taskloop grainsize(1) nogroup untied
      for (i = 0; i < 6; i++) {
        #pragma omp taskyield
        timeout_barrier_wait(&barrier, 4);
        vals[i] = 1;
      }
    }
    if (omp_get_thread_num() < 2) {
      // master does not wait the completion of taskloop.
      timeout_barrier_wait(&barrier, 4);
    }
  }

  for (i = 0; i < 6; i++) {
    if (vals[i] != 1) {
      printf("vals[%d] == %d\n", i, vals[i]);
      return 0;
    }
  }
  return 1;
}

int main() {
  int i, num_failed = 0;
  for (i = 0; i < REPETITIONS; i++) {
    if (!test_taskloop_nogroup_untied_scheduling()) {
      num_failed++;
    }
  }
  return num_failed;
}
