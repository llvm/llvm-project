// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
#include "test.h"

void *ThreadFunc(void *m) {
  AnnotateRWLockAcquired(__FILE__, __LINE__, m, 1);
  return 0;
}

int main() {
  int m = 0;
  AnnotateRWLockAcquired(__FILE__, __LINE__, &m, 1);
  pthread_t th;
  pthread_create(&th, 0, ThreadFunc, &m);
  pthread_join(th, 0);
  return 0;
}

// CHECK: WARNING: ThreadSanitizer: double lock of a mutex
// CHECK:     #0 AnnotateRWLockAcquired
// CHECK:     #1 ThreadFunc
// CHECK: Location is stack of main thread.
// CHECK:   Mutex {{.*}} created at:
// CHECK:     #0 AnnotateRWLockAcquired
// CHECK:     #1 main
// CHECK: SUMMARY: ThreadSanitizer: double lock of a mutex {{.*}}mutex_double_lock.cpp{{.*}}ThreadFunc

