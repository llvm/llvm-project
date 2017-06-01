// RUN: %clangxx_tsan %s -o %t
// RUN:                                 not %run %t 2>&1 | FileCheck %s
// RUN: %env_tsan_opts=detect_deadlocks=1 not %run %t 2>&1 | FileCheck %s
// RUN: %env_tsan_opts=detect_deadlocks=0     %run %t 2>&1 | FileCheck %s --check-prefix=DISABLED
// RUN: echo "deadlock:main" > %t.supp
// RUN: %env_tsan_opts=suppressions='%t.supp' %run %t 2>&1 | FileCheck %s --check-prefix=DISABLED
// RUN: echo "deadlock:zzzz" > %t.supp
// RUN: %env_tsan_opts=suppressions='%t.supp' not %run %t 2>&1 | FileCheck %s
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
  // CHECK: ThreadSanitizer: lock-order-inversion (potential deadlock)
  // DISABLED-NOT: ThreadSanitizer
  // DISABLED: PASS
  pthread_mutex_unlock(&mu1);
  pthread_mutex_unlock(&mu2);

  pthread_mutex_destroy(&mu1);
  pthread_mutex_destroy(&mu2);
  fprintf(stderr, "PASS\n");
}
