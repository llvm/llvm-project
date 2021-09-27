// RUN: %clangxx_tsan %darwin_min_target_with_tls_support -O1 %s -o %t && \
// RUN:   %deflake %run %t | FileCheck %s

// Race with initial TLS initialization:
// there is no explicit second write,
// but the TLS variable is published unsafely.
#include "test.h"

__thread long X;
long *P;

void *Thread(void *a) {
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
// CHECK:     #0 __tsan_tls_initialization
// CHECK:   Location is TLS of thread T1.
