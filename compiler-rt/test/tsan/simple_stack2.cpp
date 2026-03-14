// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
#include "test.h"

int Global;

void __attribute__((noinline)) foo1() {
  Global = 42;
}

void __attribute__((noinline)) bar1() {
  volatile int tmp = 42;
  int tmp2 = tmp;
  (void)tmp2;
  foo1();
}

void __attribute__((noinline)) foo2() {
  volatile int tmp = Global;
  int tmp2 = tmp;
  (void)tmp2;
}

void __attribute__((noinline)) bar2() {
  volatile int tmp = 42;
  int tmp2 = tmp;
  (void)tmp2;
  foo2();
}

void *Thread1(void *x) {
  barrier_wait(&barrier);
  bar1();
  return NULL;
}

int main() {
  barrier_init(&barrier, 2);
  pthread_t t;
  pthread_create(&t, NULL, Thread1, NULL);
  bar2();
  barrier_wait(&barrier);
  pthread_join(t, NULL);
}

// CHECK:      WARNING: ThreadSanitizer: data race
// CHECK-NEXT:   Write of size 4 at {{.*}} by thread T1:
// CHECK-NEXT:     #0 foo1{{.*}} {{.*}}simple_stack2.cpp:[[@LINE-40]]{{(:10)?}} ({{.*}})
// CHECK-NEXT:     #1 bar1{{.*}} {{.*}}simple_stack2.cpp:[[@LINE-34]]{{(:3)?}} ({{.*}})
// CHECK-NEXT:     #2 Thread1{{.*}} {{.*}}simple_stack2.cpp:[[@LINE-17]]{{(:3)?}} ({{.*}})
// CHECK:        Previous read of size 4 at {{.*}} by main thread:
// CHECK-NEXT:     #0 foo2{{.*}} {{.*}}simple_stack2.cpp:[[@LINE-33]]{{(:22)?}} ({{.*}})
// CHECK-NEXT:     #1 bar2{{.*}} {{.*}}simple_stack2.cpp:[[@LINE-25]]{{(:3)?}} ({{.*}})
// CHECK-NEXT:     #2 main{{.*}} {{.*}}simple_stack2.cpp:[[@LINE-13]]{{(:3)?}} ({{.*}})
