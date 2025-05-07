// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
#include "test.h"

void *Thread(void *a) {
  barrier_wait(&barrier);
  ((int *)a)[1] = 43;
  return 0;
}

int main() {
  barrier_init(&barrier, 2);
  int Arr[2] = {41, 42};
  pthread_t t;
  pthread_create(&t, 0, Thread, &Arr[0]);
  Arr[1] = 43;
  barrier_wait(&barrier);
  pthread_join(t, 0);
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK:   Location is stack of main thread.
