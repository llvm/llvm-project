// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
#include <pthread.h>

volatile int N;  // Prevent loop unrolling.
int **data;

void *Thread1(void *x) {
  for (int i = 0; i < N; i++)
    data[i][0] = 42;
  return 0;
}

int main() {
  N = 4;
  data = new int*[N];
  for (int i = 0; i < N; i++)
    data[i] = new int;
  pthread_t t;
  pthread_create(&t, 0, Thread1, 0);
  Thread1(0);
  pthread_join(t, 0);
  for (int i = 0; i < N; i++)
    delete data[i];
  delete[] data;
}

// CHECK: ThreadSanitizer: reported 1 warnings
