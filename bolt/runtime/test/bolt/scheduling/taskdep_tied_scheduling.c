// RUN: %libomp-compile && env KMP_ABT_NUM_ESS=4 %libomp-run
// REQUIRES: abt
#include "omp_testsuite.h"
#include "bolt_scheduling_util.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int calc_seq(int n) {
  int i, j, *buffer = (int *)malloc(sizeof(int) * n * n);
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      if (i == 0 && j == 0) {
        buffer[i * n + j] = 1;
      } else if (i == 0) {
        buffer[i * n + j] = buffer[i * n + (j - 1)];
      } else if (j == 0) {
        buffer[i * n + j] = buffer[(i - 1) * n + j];
      } else {
        buffer[i * n + j] = buffer[(i - 1) * n + j] + buffer[i * n + (j - 1)];
      }
    }
  }
  int ret = buffer[(n - 1) * n + (n - 1)];
  free(buffer);
  return ret;
}

int test_taskdep_tied_scheduilng() {
  int n = 6;
  int seq_val, task_val;

  timeout_barrier_t barrier;
  timeout_barrier_init(&barrier);

  int *A_buf = (int *)malloc(sizeof(int) * n * n);
  int **A = (int **)malloc(sizeof(int *) * n);

  #pragma omp parallel shared(task_val) firstprivate(n) num_threads(4)
  {
    #pragma omp master
    {
      // 6 ( = n) barrier_waits in diagonal tasks and 2 barrier_waits in threads
      check_num_ess(4);
      int i, j;
      for(i = 0; i < n; i++) {
        A[i] = A_buf + (i * n);
        for(j = 0; j < n; j++) {
          // Assign random values.
          A[i][j] = i * n + j;
        }
      }
      // A[i][j] is the root task.
      for(i = 0; i < n; i++) {
        for(j = 0; j < n; j++) {
          if (i == 0 && j == 0) {
            #pragma omp task depend(out:A[i][j]) firstprivate(A, i, j)
            {
              if (i + j == n - 1) {
                timeout_barrier_wait(&barrier, 4);
              }
              A[i][j] = 1;
            }
          } else if (i == 0) {
            #pragma omp task depend(in:A[i][j - 1]) depend(out:A[i][j]) \
                             firstprivate(A, i, j)
            {
              if (i + j == n - 1) {
                timeout_barrier_wait(&barrier, 4);
              }
              A[i][j] = A[i][j - 1];
            }
          } else if (j == 0) {
            #pragma omp task depend(in:A[i - 1][j]) depend(out:A[i][j]) \
                             firstprivate(A, i, j)
            {
              if (i + j == n - 1) {
                timeout_barrier_wait(&barrier, 4);
              }
              A[i][j] = A[i - 1][j];
            }
          } else {
            #pragma omp task depend(in:A[i - 1][j], A[i][j - 1]) \
                             depend(out:A[i][j])
            {
              if (i + j == n - 1) {
                timeout_barrier_wait(&barrier, 4);
              }
              A[i][j] = A[i - 1][j] + A[i][j - 1];
            }
          }
        }
      }
    }
    if (omp_get_thread_num() < 2) {
      // The master thread does not need to wait for tasks, so the master thread
      // can execute the following after task creation.
      timeout_barrier_wait(&barrier, 4);
    }
  }

  task_val = A[n - 1][n - 1];
  free(A);
  free(A_buf);

  seq_val = calc_seq(n);
  if(seq_val != task_val) {
    printf("Failed: route(%d) = %d (ANS = %d)\n", n, task_val, seq_val);
    return 0;
  }
  return 1;
}

int main() {
  int i, num_failed = 0;
  for (i = 0; i < REPETITIONS; i++) {
    if (!test_taskdep_tied_scheduilng()) {
      num_failed++;
    }
  }
  return num_failed;
}
