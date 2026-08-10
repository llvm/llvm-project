// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
#include "test.h"

_Atomic(int*) p;

void *thr(void *a) {
  barrier_wait(&barrier);
  int *pp = __c11_atomic_load(&p, __ATOMIC_RELAXED);
  *pp = 42;
  return 0;
}

int main() {
  barrier_init(&barrier, 2);
  pthread_t th;
  pthread_create(&th, 0, thr, p);
  __c11_atomic_store(&p, new int, __ATOMIC_RELAXED);
  barrier_wait(&barrier);
  pthread_join(th, 0);
}

// CHECK: data race
// CHECK:   Previous write
// CHECK:     #0 operator new
// CHECK:   Location is heap block
// CHECK:     #0 operator new
