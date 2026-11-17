// RUN: %clang_tsan -O1 %s -o %t %if freebsd %{ -lstdthreads %} && %deflake %run %t 2>&1 | FileCheck %s
// <threads.h> is missing on Darwin and before glibc 2.28.
// UNSUPPORTED: darwin, glibc && !glibc-2.28

#include "test.h"
#include <threads.h>

int Global;

int Thread1(void *x) {
  barrier_wait(&barrier);
  Global = 42;
  return 0;
}

int Thread2(void *x) {
  Global = 43;
  barrier_wait(&barrier);
  return 0;
}

int main() {
  barrier_init(&barrier, 2);
  thrd_t t[2];
  thrd_create(&t[0], Thread1, NULL);
  thrd_create(&t[1], Thread2, NULL);
  thrd_join(t[0], NULL);
  thrd_join(t[1], NULL);
  return 0;
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK:   Write of size 4 at {{.*}} by thread T1:
// CHECK:     #0 Thread1
// CHECK-NOT: __tsan_c11_thread_start_func
// CHECK:   Previous write of size 4 at {{.*}} by thread T2:
// CHECK:     #0 Thread2
// CHECK-NOT: __tsan_c11_thread_start_func
// CHECK:   Thread T1 (tid={{.*}}, running) created by main thread at:
// CHECK:     #{{[0-9]+}} thrd_create
