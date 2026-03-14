// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
#include "test.h"

pthread_mutex_t mtx;

__attribute__((noinline)) void Func1() {
  pthread_mutex_lock(&mtx);
  __tsan_check_no_mutexes_held();
  pthread_mutex_unlock(&mtx);
}

__attribute__((noinline)) void Func2() {
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
// CHECK:     {{.*}}__tsan_check_no_mutexes_held{{.*}}
// CHECK:     {{.*}}Func1{{.*}}
// CHECK:     {{.*}}main{{.*}}
// CHECK:   Mutex {{.*}} created at:
// CHECK:     {{.*}}pthread_mutex_init{{.*}}
// CHECK:     {{.*}}main{{.*}}
// CHECK: SUMMARY: ThreadSanitizer: mutex held in the wrong context {{.*}}mutex_held_wrong_context.cpp{{.*}}Func1

// CHECK-NOT: SUMMARY: ThreadSanitizer: mutex held in the wrong context {{.*}}mutex_held_wrong_context.cpp{{.*}}Func2
