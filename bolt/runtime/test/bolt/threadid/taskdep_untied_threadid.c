// RUN: %libomp-compile-and-run
// REQUIRES: abt && !clang

// Clang 10.0 discards local variables saved before taskyield.  We mark untied
// task tests that use local variables across taskyield with Clang as
// unsupported so far.
#include "omp_testsuite.h"
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

#define TASK_UNTIED_CHECK(_val_index)                                          \
  do {                                                                         \
    int val_index = (_val_index);                                              \
    ABT_thread abt_thread;                                                     \
    ABT_EXIT_IF_FAIL(ABT_thread_self(&abt_thread));                            \
                                                                               \
    _Pragma("omp taskyield")                                                   \
                                                                               \
    int omp_thread_id2 = omp_get_thread_num();                                 \
    ABT_thread abt_thread2;                                                    \
    ABT_EXIT_IF_FAIL(ABT_thread_self(&abt_thread2));                           \
    ABT_bool abt_thread_equal;                                                 \
    ABT_EXIT_IF_FAIL(ABT_thread_equal(abt_thread, abt_thread2,                 \
                                      &abt_thread_equal));                     \
    if (abt_thread_equal == ABT_TRUE) {                                        \
      vals[val_index] += 1;                                                    \
    }                                                                          \
                                                                               \
    ABT_EXIT_IF_FAIL(ABT_thread_yield());                                      \
                                                                               \
    int omp_thread_id3 = omp_get_thread_num();                                 \
    if (omp_thread_id2 == omp_thread_id3) {                                    \
      vals[val_index] += 2;                                                    \
    }                                                                          \
  } while (0)

int test_taskdep_untied_threadid(int num_threads) {
  int n = 10;
  int seq_val, task_val;

  int vals[n * n];
  memset(vals, 0, sizeof(int) * n * n);

  #pragma omp parallel shared(task_val) firstprivate(n) num_threads(num_threads)
  #pragma omp master
  {
    int i, j;
    int *A_buf = (int *)malloc(sizeof(int) * n * n);
    int **A = (int **)malloc(sizeof(int *) * n);
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
          #pragma omp task depend(out:A[i][j]) firstprivate(A, i, j) untied
          {
            TASK_UNTIED_CHECK(i * n + j);
            A[i][j] = 1;
          }
        } else if (i == 0) {
          #pragma omp task depend(in:A[i][j - 1]) depend(out:A[i][j]) \
                           firstprivate(A, i, j) untied
          {
            TASK_UNTIED_CHECK(i * n + j);
            A[i][j] = A[i][j - 1];
          }
        } else if (j == 0) {
          #pragma omp task depend(in:A[i - 1][j]) depend(out:A[i][j]) \
                           firstprivate(A, i, j) untied
          {
            TASK_UNTIED_CHECK(i * n + j);
            A[i][j] = A[i - 1][j];
          }
        } else {
          #pragma omp task depend(in:A[i - 1][j], A[i][j - 1]) \
                           depend(out:A[i][j]) untied
          {
            TASK_UNTIED_CHECK(i * n + j);
            A[i][j] = A[i - 1][j] + A[i][j - 1];
          }
        }
      }
    }
    #pragma omp taskwait
    task_val = A[n - 1][n - 1];
    free(A);
    free(A_buf);
  }

  seq_val = calc_seq(n);
  if(seq_val != task_val) {
    printf("[%d] Failed: route(%d) = %d (ANS = %d)\n", num_threads, n, task_val,
           seq_val);
    return 0;
  }
  int index;
  for (index = 0; index < n * n; index++) {
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
    if (!test_taskdep_untied_threadid(i + 1)) {
      num_failed++;
    }
  }
  return num_failed;
}
