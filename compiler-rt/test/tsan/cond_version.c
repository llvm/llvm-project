// RUN: %clang_tsan -O1 %s -o %t -lrt && %run %t 2>&1 | FileCheck %s
// Test that pthread_cond is properly intercepted,
// previously there were issues with versioned symbols.
// CHECK: OK

// OS X doesn't have pthread_condattr_setclock.
// UNSUPPORTED: darwin

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <errno.h>

int main() {
  typedef unsigned long long u64;
  pthread_mutex_t m;
  pthread_cond_t c;
  pthread_condattr_t at;
  struct timespec ts0, ts1, ts2;
  int res;
  u64 sleep;

  pthread_mutex_init(&m, 0);
  pthread_condattr_init(&at);
  pthread_condattr_setclock(&at, CLOCK_MONOTONIC);
  pthread_cond_init(&c, &at);

  clock_gettime(CLOCK_MONOTONIC, &ts0);
  ts1 = ts0;
  ts1.tv_sec += 2;

  pthread_mutex_lock(&m);
  do {
    res = pthread_cond_timedwait(&c, &m, &ts1);
  } while (res == 0);
  pthread_mutex_unlock(&m);

  clock_gettime(CLOCK_MONOTONIC, &ts2);
  sleep = (u64)ts2.tv_sec * 1000000000 + ts2.tv_nsec -
      ((u64)ts0.tv_sec * 1000000000 + ts0.tv_nsec);
  if (res != ETIMEDOUT)
    exit(printf("bad return value %d, want %d\n", res, ETIMEDOUT));
  if (sleep < 1000000000)
    exit(printf("bad sleep duration %lluns, want %dns\n", sleep, 1000000000));
  fprintf(stderr, "OK\n");
}
