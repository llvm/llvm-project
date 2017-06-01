// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s --check-prefix=CHECK-%os --check-prefix=CHECK
#include "test.h"

void *Thread(void *a) {
  barrier_wait(&barrier);
  *(int*)a = 43;
  return 0;
}

int main() {
  barrier_init(&barrier, 2);
  static __thread int Var = 42;
  pthread_t t;
  pthread_create(&t, 0, Thread, &Var);
  Var = 43;
  barrier_wait(&barrier);
  pthread_join(t, 0);
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK-Linux:   Location is TLS of main thread.
// CHECK-FreeBSD:   Location is TLS of main thread.
// CHECK-Darwin:   Location is heap block of size 4
