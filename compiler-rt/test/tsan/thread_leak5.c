// RUN: %clang_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
#include "test.h"

void *Thread(void *x) {
  barrier_wait(&barrier);
  return 0;
}

int main() {
  volatile int N = 5;  // prevent loop unrolling
  barrier_init(&barrier, N + 1);
  for (int i = 0; i < N; i++) {
    pthread_t t;
    pthread_create(&t, 0, Thread, 0);
  }
  barrier_wait(&barrier);
  sleep(1);  // wait for the threads to finish and exit
  return 0;
}

// CHECK: WARNING: ThreadSanitizer: thread leak
// CHECK:   And 4 more similar thread leaks
