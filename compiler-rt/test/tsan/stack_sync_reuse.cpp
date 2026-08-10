// RUN: %clang_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
#include "test.h"

// Test case https://github.com/google/sanitizers/issues/494
// Tsan sees false HB edge on address pointed to by syncp variable.
// It is false because when acquire is done syncp points to a var in one frame,
// and during release it points to a var in a different frame.
// The code is somewhat tricky because it prevents compiler from optimizing
// our accesses away, structured to not introduce other data races and
// not introduce other synchronization, and to arrange the vars in different
// frames to occupy the same address.

// The data race CHECK-NOT below actually must be CHECK, because the program
// does contain the data race on global.

// CHECK-NOT: WARNING: ThreadSanitizer: data race
// CHECK: DONE

long global;
long *syncp;
long *addr;
long sink;

void *Thread(void *x) {
  while (__atomic_load_n(&syncp, __ATOMIC_ACQUIRE) == 0)
    usleep(1000);  // spin wait
  global = 42;
  __atomic_store_n(syncp, 1, __ATOMIC_RELEASE);
  __atomic_store_n(&syncp, 0, __ATOMIC_RELAXED);
  return NULL;
}

void __attribute__((noinline)) foobar() {
  __attribute__((aligned(64))) long s;

  addr = &s;
  __atomic_store_n(&s, 0, __ATOMIC_RELAXED);
  __atomic_store_n(&syncp, &s, __ATOMIC_RELEASE);
  while (__atomic_load_n(&syncp, __ATOMIC_RELAXED) != 0)
    usleep(1000);  // spin wait
}

void __attribute__((noinline)) barfoo() {
  __attribute__((aligned(64))) long s;

  if (addr != &s) {
    printf("address mismatch addr=%p &s=%p\n", addr, &s);
    exit(1);
  }
  __atomic_store_n(&addr, &s, __ATOMIC_RELAXED);
  __atomic_store_n(&s, 0, __ATOMIC_RELAXED);
  sink = __atomic_load_n(&s, __ATOMIC_ACQUIRE);
  global = 43;
}

int main() {
  pthread_t t;
  pthread_create(&t, 0, Thread, 0);
  foobar();
  barfoo();
  pthread_join(t, 0);
  if (sink != 0)
    exit(1);
  fprintf(stderr, "DONE\n");
  return 0;
}

