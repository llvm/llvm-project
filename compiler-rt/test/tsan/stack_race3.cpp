// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s

// Race with initial stack initialization:
// there is no explicit second write,
// but the stack variable is published unsafely.
#include "test.h"

long *P;

void *Thread(void *a) {
  long X;
  __atomic_store_n(&P, &X, __ATOMIC_RELAXED);
  barrier_wait(&barrier);
  barrier_wait(&barrier);
  return 0;
}

int main() {
  barrier_init(&barrier, 2);
  pthread_t t;
  pthread_create(&t, NULL, Thread, NULL);
  barrier_wait(&barrier);
  long *p = __atomic_load_n(&P, __ATOMIC_RELAXED);
  *p = 42;
  barrier_wait(&barrier);
  pthread_join(t, 0);
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK:   Write of size 8 at {{.*}} by main thread:
// CHECK:     #0 main
// CHECK:   Previous write of size 8 at {{.*}} by thread T1:
// CHECK:     #0 __tsan_stack_initialization
// CHECK:   Location is stack of thread T1
