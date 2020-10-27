// RUN: %compile-run-and-check

#include <omp.h>
#include <stdio.h>

#pragma omp declare target
static void putValueInParallel(int *ptr, int value) {
  #pragma omp parallel
  {
    *ptr = value;
  }
}

static int getId() {
  int id;
  putValueInParallel(&id, omp_get_thread_num());
  return id;
}
#pragma omp end declare target

const int MaxThreads = 1024;
const int Threads = 64;

int main(int argc, char *argv[]) {
  int master;
  int check[MaxThreads];
  for (int i = 0; i < MaxThreads; i++) {
    check[i] = 0;
  }

  #pragma omp target map(master, check[:])
  {
    master = getId();

    #pragma omp parallel num_threads(Threads)
    {
      check[omp_get_thread_num()] = getId();
    }
  }

  // CHECK: master = 0.
  printf("master = %d.\n", master);
  // CHECK-NOT: invalid
  for (int i = 0; i < MaxThreads; i++) {
    if (i < Threads) {
      if (check[i] != i) {
        printf("invalid: check[%d] should be %d, is %d\n", i, i, check[i]);
      }
    } else if (check[i] != 0) {
      printf("invalid: check[%d] should be 0, is %d\n", i, check[i]);
    }
  }

  return 0;
}
