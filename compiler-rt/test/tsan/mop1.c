// RUN: %clang_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
#include "test.h"

// We want to establish the following sequence of accesses to X:
// - main thread writes X
// - thread2 reads X, this read happens-before the write in main thread
// - thread1 reads X, this read is concurrent with the write in main thread
// Write in main thread and read in thread1 should be detected as a race.
// Previously tsan replaced write by main thread with read by thread1,
// as the result the race was not detected.

volatile long X, Y, Z;

void *Thread1(void *x) {
  barrier_wait(&barrier);
  barrier_wait(&barrier);
  Y = X;
  return NULL;
}

void *Thread2(void *x) {
  Z = X;
  barrier_wait(&barrier);
  return NULL;
}

int main() {
  barrier_init(&barrier, 2);
  pthread_t t[2];
  pthread_create(&t[0], 0, Thread1, 0);
  X = 42;
  barrier_wait(&barrier);
  pthread_create(&t[1], 0, Thread2, 0);
  pthread_join(t[0], 0);
  pthread_join(t[1], 0);
  return 0;
}

// CHECK: WARNING: ThreadSanitizer: data race

