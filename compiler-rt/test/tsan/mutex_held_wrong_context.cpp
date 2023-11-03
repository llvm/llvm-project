// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
#include "test.h"

pthread_mutex_t mtx;

void Func1() {
  pthread_mutex_lock(&mtx);
  __tsan_check_no_mutexes_held();
  pthread_mutex_unlock(&mtx);
}

void Func2() {
  pthread_mutex_lock(&mtx);
  pthread_mutex_unlock(&mtx);
  __tsan_check_no_mutexes_held();
}

int main() {
  pthread_mutex_init(&mtx, NULL);
  Func1();
  Func2();
  return 0;
}

// CHECK: WARNING: ThreadSanitizer: mutex held in the wrong context
// CHECK:     #0 __tsan_check_no_mutexes_held
// CHECK:     #1 Func1
// CHECK:     #2 main
// CHECK:   Mutex {{.*}} created at:
// CHECK:     #0 pthread_mutex_init
// CHECK:     #1 main
// CHECK: SUMMARY: ThreadSanitizer: mutex held in the wrong context {{.*}}mutex_held_wrong_context.cpp{{.*}}Func1

// CHECK-NOT: SUMMARY: ThreadSanitizer: mutex held in the wrong context {{.*}}mutex_held_wrong_context.cpp{{.*}}Func2
