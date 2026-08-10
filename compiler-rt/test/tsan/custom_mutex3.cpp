// RUN: %clangxx_tsan -O1 --std=c++11 %s -o %t
// RUN: %env_tsan_opts=report_destroy_locked=0 %run %t 2>&1 | FileCheck %s
#include "custom_mutex.h"

// Regression test for a bug.
// Thr1 destroys a locked mutex, previously such mutex was not removed from
// sync map and as the result subsequent uses of a mutex located at the same
// address caused false race reports.

Mutex mu(false, __tsan_mutex_write_reentrant);
long data;

void *thr1(void *arg) {
  mu.Lock();
  mu.~Mutex();
  new(&mu) Mutex(true, __tsan_mutex_write_reentrant);
  return 0;
}

void *thr2(void *arg) {
  barrier_wait(&barrier);
  mu.Lock();
  data++;
  mu.Unlock();
  return 0;
}

int main() {
  barrier_init(&barrier, 2);
  pthread_t th;
  pthread_create(&th, 0, thr1, 0);
  pthread_join(th, 0);

  barrier_init(&barrier, 2);
  pthread_create(&th, 0, thr2, 0);
  mu.Lock();
  data++;
  mu.Unlock();
  barrier_wait(&barrier);
  pthread_join(th, 0);
  fprintf(stderr, "DONE\n");
  return 0;
}

// CHECK-NOT: WARNING: ThreadSanitizer: data race
// CHECK: DONE
