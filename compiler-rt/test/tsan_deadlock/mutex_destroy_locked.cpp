// Equivalent to tsan/mutex_destroy_locked.cpp
// RUN: %clangxx_tsan_deadlock -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s

#include <pthread.h>

int main() {
  pthread_mutex_t m;
  pthread_mutex_init(&m, 0);
  pthread_mutex_lock(&m);
  pthread_mutex_destroy(&m);
  return 0;
}

// CHECK: WARNING: DeadlockSanitizer: destroy of a locked mutex
// CHECK:     {{.*}} in pthread_mutex_destroy
// CHECK:   and:
// CHECK:     {{.*}} in pthread_mutex_lock
// CHECK:   Mutex M0 (0x{{.*}}) created at:
// CHECK:     {{.*}} in pthread_mutex_init
// CHECK: SUMMARY: DeadlockSanitizer: destroy of a locked mutex
