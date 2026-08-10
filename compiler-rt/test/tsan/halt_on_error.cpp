// RUN: %clang_tsan -O1 %s -o %t && %env_tsan_opts=halt_on_error=1 %deflake %run %t | FileCheck %s
#include "test.h"

int X;

void *Thread(void *x) {
  barrier_wait(&barrier);
  X = 42;
  return 0;
}

int main() {
  barrier_init(&barrier, 2);
  fprintf(stderr, "BEFORE\n");
  pthread_t t;
  pthread_create(&t, 0, Thread, 0);
  X = 43;
  barrier_wait(&barrier);
  pthread_join(t, 0);
  fprintf(stderr, "AFTER\n");
  return 0;
}

// CHECK: BEFORE
// CHECK: WARNING: ThreadSanitizer: data race
// CHECK-NOT: AFTER

