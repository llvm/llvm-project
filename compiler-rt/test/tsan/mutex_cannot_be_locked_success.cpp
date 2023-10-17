// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
#include "test.h"

pthread_mutex_t mtx;

void *ThreadFunc(void *) {
  pthread_mutex_lock(&mtx);
  pthread_mutex_unlock(&mtx);
  __tsan_check_no_mutexes_held();
}

int main() {
  pthread_t th;
  pthread_create(&th, 0, ThreadFunc, NULL);
  pthread_join(th, 0);
  return 0;
}

// CHECK-NOT: WARNING: ThreadSanitizer: mutex cannot be locked on this code path
