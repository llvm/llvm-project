// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s --check-prefix=CHECK-%os --check-prefix=CHECK
#include "test.h"

void *Thread2(void *a) {
  barrier_wait(&barrier);
  *(int*)a = 43;
  return 0;
}

void *Thread(void *a) {
  static __thread int Var = 42;
  pthread_t t;
  pthread_create(&t, 0, Thread2, &Var);
  Var = 42;
  barrier_wait(&barrier);
  pthread_join(t, 0);
  return 0;
}

int main() {
  barrier_init(&barrier, 2);
  pthread_t t;
  pthread_create(&t, 0, Thread, 0);
  pthread_join(t, 0);
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK-Linux:   Location is TLS of thread T1.
// CHECK-FreeBSD:   Location is TLS of thread T1.
// CHECK-Darwin:   Location is heap block of size 4
