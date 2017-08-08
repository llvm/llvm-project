#include "test.h"

int Global;

void __attribute__((noinline)) foo1() {
  Global = 42;
}

void __attribute__((noinline)) bar1() {
  volatile int tmp = 42; (void)tmp;
  foo1();
}

void __attribute__((noinline)) foo2() {
  volatile int v = Global; (void)v;
}

void __attribute__((noinline)) bar2() {
  volatile int tmp = 42; (void)tmp;
  foo2();
}

void *Thread1(void *x) {
  barrier_wait(&barrier);
  bar1();
  return NULL;
}

void *Thread2(void *x) {
  bar2();
  barrier_wait(&barrier);
  return NULL;
}

void StartThread(pthread_t *t, void *(*f)(void*)) {
  pthread_create(t, NULL, f, NULL);
}

int main() {
  barrier_init(&barrier, 2);
  pthread_t t[2];
  StartThread(&t[0], Thread1);
  StartThread(&t[1], Thread2);
  pthread_join(t[0], NULL);
  pthread_join(t[1], NULL);
  return 0;
}

// RUN: %clang_tsan -O1 %s -o %t && %deflake %run %t 2>&1 | FileCheck %s

// Also check that functions instrumentation can be configured by either driver
// or legacy flags:

// RUN: %clangxx_tsan -O1 %s -o %t -fno-sanitize-thread-func-entry-exit && %deflake %run %t 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FUNC-ENTRY-EXIT-OFF %s
// RUN: %clangxx_tsan -O1 %s -o %t -mllvm -tsan-instrument-func-entry-exit=0 && %deflake %run %t 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FUNC-ENTRY-EXIT-OFF %s

// CHECK:      WARNING: ThreadSanitizer: data race
// CHECK-NEXT:   Write of size 4 at {{.*}} by thread T1:
// CHECK-NEXT:     #0 foo1{{.*}} {{.*}}simple_stack.c:6{{(:10)?}} ({{.*}})
// CHECK-NEXT:     #1 bar1{{.*}} {{.*}}simple_stack.c:11{{(:3)?}} ({{.*}})
// CHECK-NEXT:     #2 Thread1{{.*}} {{.*}}simple_stack.c:25{{(:3)?}} ({{.*}})
// CHECK:        Previous read of size 4 at {{.*}} by thread T2:
// CHECK-NEXT:     #0 foo2{{.*}} {{.*}}simple_stack.c:15{{(:20)?}} ({{.*}})
// CHECK-NEXT:     #1 bar2{{.*}} {{.*}}simple_stack.c:20{{(:3)?}} ({{.*}})
// CHECK-NEXT:     #2 Thread2{{.*}} {{.*}}simple_stack.c:30{{(:3)?}} ({{.*}})
// CHECK:        Thread T1 (tid={{.*}}, running) created by main thread at:
// CHECK-NEXT:     #0 pthread_create {{.*}} ({{.*}})
// CHECK-NEXT:     #1 StartThread{{.*}} {{.*}}simple_stack.c:36{{(:3)?}} ({{.*}})
// CHECK-NEXT:     #2 main{{.*}} {{.*}}simple_stack.c:42{{(:3)?}} ({{.*}})
// CHECK:        Thread T2 ({{.*}}) created by main thread at:
// CHECK-NEXT:     #0 pthread_create {{.*}} ({{.*}})
// CHECK-NEXT:     #1 StartThread{{.*}} {{.*}}simple_stack.c:36{{(:3)?}} ({{.*}})
// CHECK-NEXT:     #2 main{{.*}} {{.*}}simple_stack.c:43{{(:3)?}} ({{.*}})

// CHECK-FUNC-ENTRY-EXIT-OFF:      WARNING: ThreadSanitizer: data race
// CHECK-FUNC-ENTRY-EXIT-OFF-NEXT:   Write of size 4 at {{.*}} by thread T1:
// CHECK-FUNC-ENTRY-EXIT-OFF-NEXT:     #0 foo1{{.*}} {{.*}}simple_stack.c:6{{(:10)?}} ({{.*}})
// CHECK-FUNC-ENTRY-EXIT-OFF:        Previous read of size 4 at {{.*}} by thread T2:
// CHECK-FUNC-ENTRY-EXIT-OFF-NEXT:     #0 foo2{{.*}} {{.*}}simple_stack.c:15{{(:20)?}} ({{.*}})
// CHECK-FUNC-ENTRY-EXIT-OFF:        Thread T1 (tid={{.*}}, running) created by main thread at:
// CHECK-FUNC-ENTRY-EXIT-OFF-NEXT:     #0 pthread_create {{.*}} ({{.*}})
// CHECK-FUNC-ENTRY-EXIT-OFF:        Thread T2 ({{.*}}) created by main thread at:
// CHECK-FUNC-ENTRY-EXIT-OFF-NEXT:     #0 pthread_create {{.*}} ({{.*}})
