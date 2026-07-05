// RUN: %clangxx_tsan %darwin_min_target_with_tls_support -O1 %s -o %t && \
// RUN:   %deflake %run %t | \
// RUN:   FileCheck %s --check-prefix=CHECK-%os --check-prefix=CHECK
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
// CHECK-NetBSD:   Location is TLS of main thread.
// CHECK-Darwin:   Location is heap block of size 4
