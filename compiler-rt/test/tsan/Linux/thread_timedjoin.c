// RUN: %clang_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
#define _GNU_SOURCE
#include "../test.h"
#include <errno.h>

int var;

void *Thread(void *x) {
  barrier_wait(&barrier);
  var = 1;
  return 0;
}

static void check(int res, int expect) {
  if (res != expect) {
    fprintf(stderr, "Unexpected result of pthread_timedjoin_np: %d\n", res);
    exit(1);
  }
}

int main() {
  barrier_init(&barrier, 2);
  pthread_t t;
  pthread_create(&t, 0, Thread, 0);
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  check(pthread_timedjoin_np(t, 0, &ts), ETIMEDOUT);
  barrier_wait(&barrier);
  clock_gettime(CLOCK_REALTIME, &ts);
  ts.tv_sec += 10000;
  check(pthread_timedjoin_np(t, 0, &ts), 0);
  var = 2;
  fprintf(stderr, "PASS\n");
  return 0;
}

// CHECK-NOT: WARNING: ThreadSanitizer: data race
// CHECK-NOT: WARNING: ThreadSanitizer: thread leak
// CHECK: PASS
