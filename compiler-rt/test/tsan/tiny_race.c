// RUN: %clang_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
#include "test.h"

int Global;

void *Thread1(void *x) {
  barrier_wait(&barrier);
  Global = 42;
  return x;
}

int main() {
  barrier_init(&barrier, 2);
  pthread_t t;
  pthread_create(&t, 0, Thread1, 0);
  Global = 43;
  barrier_wait(&barrier);
  pthread_join(t, 0);
  return Global;
}

// CHECK: WARNING: ThreadSanitizer: data race
