// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
// Regtest for https://github.com/google/sanitizers/issues/447
// This is a correct program and tsan should not report a race.
#include "test.h"

int g;
__attribute__((noinline))
int foo(int cond) {
  if (cond)
    return g;
  return 0;
}

void *Thread1(void *p) {
  barrier_wait(&barrier);
  long res = foo((long)p);
  return (void*) res;
}

int main() {
  barrier_init(&barrier, 2);
  pthread_t t;
  pthread_create(&t, 0, Thread1, 0);
  g = 1;
  barrier_wait(&barrier);
  pthread_join(t, 0);
  fprintf(stderr, "PASS\n");
  // CHECK-NOT: ThreadSanitizer: data race
  // CHECK: PASS
}
