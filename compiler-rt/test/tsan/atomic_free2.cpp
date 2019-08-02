// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
#include "test.h"

void *Thread(void *a) {
  barrier_wait(&barrier);
  __atomic_fetch_add((int*)a, 1, __ATOMIC_SEQ_CST);
  return 0;
}

int main() {
  barrier_init(&barrier, 2);
  int *a = new int(0);
  pthread_t t;
  pthread_create(&t, 0, Thread, a);
  delete a;
  barrier_wait(&barrier);
  pthread_join(t, 0);
}

// CHECK: WARNING: ThreadSanitizer: heap-use-after-free
