// RUN: %libomp-compile-and-run
#include <stdio.h>
#include <stdlib.h>
#include "omp_testsuite.h"

int calc_seq(int n) {
  int i, j, ret;
  int *buffer = (int *)malloc(sizeof(int) * n * n);
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
  ret = buffer[(n - 1) * n + (n - 1)];
  free(buffer);
  return ret;
}

int main()
{
  int r;
  int n = 5;
  int num_failed=0;

  for(r = 0; r < REPETITIONS; r++) {
    int seq_val, task_val;
    #pragma omp parallel shared(task_val) firstprivate(n)
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
            #pragma omp task depend(out:A[i][j]) firstprivate(A, i, j)
            {
              A[i][j] = 1;
            }
          } else if (i == 0) {
            #pragma omp task depend(in:A[i][j - 1]) depend(out:A[i][j])\
                             firstprivate(A, i, j)
            {
              A[i][j] = A[i][j - 1];
            }
          } else if (j == 0) {
            #pragma omp task depend(in:A[i - 1][j]) depend(out:A[i][j])\
                             firstprivate(A, i, j)
            {
              A[i][j] = A[i - 1][j];
            }
          } else {
            #pragma omp task depend(in:A[i - 1][j], A[i][j - 1])\
                             depend(out:A[i][j])
            {
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
      printf("[%d] Failed: route(%d) = %d (ANS = %d)\n", r, n, task_val,
             seq_val);
      num_failed++;
    }
  }
  return num_failed;
}
