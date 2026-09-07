// Equivalent to tsan/mutex_cycle2.c
// RUN: %clangxx_tsan_deadlock %s -o %t
// RUN:                                         not %run %t 2>&1 | FileCheck %s
// RUN: %env_tsan_deadlock_opts=report_bugs=0       %run %t 2>&1 | FileCheck %s --check-prefix=DISABLED
#include <pthread.h>
#include <stdio.h>

int main() {
  pthread_mutex_t mu1, mu2;
  pthread_mutex_init(&mu1, NULL);
  pthread_mutex_init(&mu2, NULL);

  // mu1 => mu2
  pthread_mutex_lock(&mu1);
  pthread_mutex_lock(&mu2);
  pthread_mutex_unlock(&mu2);
  pthread_mutex_unlock(&mu1);

  // mu2 => mu1
  pthread_mutex_lock(&mu2);
  pthread_mutex_lock(&mu1);
  // CHECK: DeadlockSanitizer: lock-order-inversion (potential deadlock)
  // DISABLED-NOT: DeadlockSanitizer
  // DISABLED: PASS
  pthread_mutex_unlock(&mu1);
  pthread_mutex_unlock(&mu2);

  pthread_mutex_destroy(&mu1);
  pthread_mutex_destroy(&mu2);
  fprintf(stderr, "PASS\n");
}
