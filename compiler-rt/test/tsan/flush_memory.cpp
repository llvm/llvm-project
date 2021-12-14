// RUN: %clangxx_tsan -O1 %s -o %t
// RUN: %env_tsan_opts=flush_memory_ms=1:flush_symbolizer_ms=1:memory_limit_mb=1 not %run %t 2>&1 | FileCheck %s
#include "test.h"

long X, Y;

void *Thread(void *arg) {
  __atomic_fetch_add(&X, 1, __ATOMIC_SEQ_CST);
  barrier_wait(&barrier);
  barrier_wait(&barrier);
  Y = 1;
  return &Y;
}

int main() {
  __tsan_flush_memory();
  barrier_init(&barrier, 2);
  __atomic_fetch_add(&X, 1, __ATOMIC_SEQ_CST);
  pthread_t t;
  pthread_create(&t, NULL, Thread, NULL);
  barrier_wait(&barrier);
  __tsan_flush_memory();
  // Trigger a race to test flushing of the symbolizer cache.
  Y = 2;
  barrier_wait(&barrier);
  pthread_join(t, NULL);
  __atomic_fetch_add(&X, 1, __ATOMIC_SEQ_CST);
  // Background runtime thread should do some flushes meanwhile.
  sleep(2);
  __tsan_flush_memory();
  fprintf(stderr, "DONE\n");
  // The race may not be detected since we are doing aggressive flushes
  // (if the state flush happens between racing accesses, tsan won't
  // detect the race). So return 1 to make the test deterministic.
  return 1;
}

// CHECK: DONE
