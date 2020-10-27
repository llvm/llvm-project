// RUN: %libomp-compile-and-run
#include <stdio.h>
#include "omp_testsuite.h"

int fib(int n) {
  int a, b;
  if (n < 2) {
    return n;
  } else {
    if(n < 4) {
      return fib(n - 1) + fib(n - 2);
    } else {
      #pragma omp task shared(a)
      {
        a = fib(n - 1);
      }
      #pragma omp task shared(b)
      {
        b = fib(n - 2);
      }
      #pragma omp taskwait
      return a + b;
    }
  }
}

int fib_seq(int n) {
  int a, b;
  if (n < 2) {
    return n;
  } else {
    a = fib_seq(n - 1);
    b = fib_seq(n - 2);
    return a + b;
  }
}

int main() {
  int i;
  int n = 20;
  int num_failed = 0;

  for(i = 0; i < REPETITIONS; i++) {
    int task_val = 0;
    int seq_val = fib_seq(n);
    #pragma omp parallel shared(task_val) firstprivate(n)
    #pragma omp master
    {
      task_val = fib(n);
    }
    if(seq_val != task_val) {
      printf("[%d] Failed: fib(%d) = %d (ANS = %d)\n", i, n, task_val, seq_val);
      num_failed++;
    }
  }
  return num_failed;
}
