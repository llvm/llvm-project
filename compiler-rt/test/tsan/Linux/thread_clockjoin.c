// RUN: %clang_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
// REQUIRES: glibc-2.31

#define _GNU_SOURCE
#include "../test.h"
#include <assert.h>
#include <errno.h>
#include <time.h>

int var;

void *Thread(void *arg) {
  barrier_wait(&barrier);
  var = 1;
  return arg;
}

int main() {
  const clockid_t clocks[] = {CLOCK_REALTIME, CLOCK_MONOTONIC};
  for (unsigned c = 0; c < sizeof(clocks) / sizeof(clocks[0]); ++c) {
    // Repeated joins allow libc to reuse pthread_t values. A successful join
    // must remove the old user ID from TSan's thread registry.
    for (int i = 0; i < 10; ++i) {
      barrier_init(&barrier, 2);
      pthread_t t;
      assert(pthread_create(&t, 0, Thread, &var) == 0);
      struct timespec ts;
      assert(clock_gettime(clocks[c], &ts) == 0);
      // The thread cannot exit until we release the barrier. A failed join
      // must preserve the user ID so that the subsequent join can find it.
      assert(pthread_clockjoin_np(t, 0, clocks[c], &ts) == ETIMEDOUT);
      barrier_wait(&barrier);
      assert(clock_gettime(clocks[c], &ts) == 0);
      ts.tv_sec += 10000;
      void *ret;
      assert(pthread_clockjoin_np(t, &ret, clocks[c], &ts) == 0);
      assert(ret == &var);
      // This write races with Thread unless the successful join acquires its
      // clock. The test barrier deliberately provides no TSan synchronization.
      var = 2;
    }
  }
  fprintf(stderr, "PASS\n");
  return 0;
}

// CHECK-NOT: WARNING: ThreadSanitizer:
// CHECK: PASS
// CHECK-NOT: WARNING: ThreadSanitizer:
