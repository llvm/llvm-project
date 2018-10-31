// RUN: %clangxx_tsan %s -o %t
// RUN: %deflake %run %t | FileCheck %s
// RUN: %deflake %run %t 1 | FileCheck %s

// The pthread_mutex_lock interceptor assumes incompatible internals w/ NetBSD
// XFAIL: netbsd

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  pthread_mutex_t *m = (pthread_mutex_t *)malloc(sizeof(pthread_mutex_t));
  pthread_mutex_init(m, 0);
  pthread_mutex_lock(m);
  pthread_mutex_unlock(m);
  pthread_mutex_destroy(m);

  if (argc > 1 && argv[1][0] == '1')
    free(m);

  pthread_mutex_lock(m);
  // CHECK: WARNING: ThreadSanitizer: use of an invalid mutex (e.g. uninitialized or destroyed)
  // CHECK:   #0 pthread_mutex_lock
  // CHECK:   #1 main {{.*}}mutex_lock_destroyed.cc:[[@LINE-3]]

  return 0;
}
