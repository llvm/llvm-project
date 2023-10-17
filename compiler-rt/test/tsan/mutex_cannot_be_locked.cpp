// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
#include "test.h"

pthread_mutex_t mtx;

void *ThreadFunc(void *) {
  pthread_mutex_lock(&mtx);
  __tsan_check_no_mutexes_held();
}

int main() {
  pthread_t th;
  pthread_create(&th, 0, ThreadFunc, NULL);
  pthread_join(th, 0);
  return 0;
}

// CHECK: WARNING: ThreadSanitizer: mutex cannot be locked on this code path
// CHECK:     #0 __tsan_check_no_mutexes_held
// CHECK:     #1 ThreadFunc
// CHECK:   Mutex {{.*}} acquired at:
// CHECK:     #0 pthread_mutex_lock
// CHECK:     #1 ThreadFunc
// CHECK: SUMMARY: ThreadSanitizer: mutex cannot be locked on this code path {{.*}}mutex_cannot_be_locked.cpp{{.*}}ThreadFunc
