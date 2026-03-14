// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
#include "test.h"

volatile int X;

void *Thread1(void *x) {
  barrier_wait(&barrier);
  X = 42;
  X = 66;
  X = 78;
  return 0;
}

void *Thread2(void *x) {
  X = 11;
  X = 99;
  X = 73;
  barrier_wait(&barrier);
  return 0;
}

int main() {
  barrier_init(&barrier, 2);
  pthread_t t;
  pthread_create(&t, 0, Thread1, 0);
  Thread2(0);
  pthread_join(t, 0);
}

// CHECK: ThreadSanitizer: reported 1 warnings
