// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
#include "test.h"

// Test for https://github.com/google/sanitizers/issues/602

void *Thread(void *a) {
  __atomic_store_n((int*)a, 1, __ATOMIC_RELAXED);
  return 0;
}

int main() {
  int *a = new int(0);
  pthread_t t;
  pthread_create(&t, 0, Thread, a);
  while (__atomic_load_n(a, __ATOMIC_RELAXED) == 0)
    sched_yield();
  delete a;
  pthread_join(t, 0);
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK:   Write
// CHECK:     #0 operator delete
// CHECK:     #1 main

// CHECK:   Previous atomic write
// CHECK:     #0 Thread
