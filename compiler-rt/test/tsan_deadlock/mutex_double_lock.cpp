// Adapted from tsan/mutex_double_lock.cpp
// RUN: %clangxx_tsan_deadlock -O1 %s -o %t && %env_tsan_deadlock_opts=halt_on_error=1 not %run %t 2>&1 | FileCheck %s

#include <pthread.h>

int main() {
  pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
  pthread_mutex_lock(&mu);
  pthread_mutex_lock(&mu);
  return 0;
}

// CHECK: WARNING: DeadlockSanitizer: double lock of a mutex
// CHECK:     {{.*}} in pthread_mutex_lock
// CHECK:   Mutex M0 (0x{{.*}}) created at:
// CHECK:     {{.*}} in pthread_mutex_lock
// CHECK: SUMMARY: DeadlockSanitizer: double lock of a mutex
