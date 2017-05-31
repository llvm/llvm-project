// RUN: %clangxx_tsan -O1 %s -o %T/simple_stack2.cc.exe && %deflake %run %T/simple_stack2.cc.exe | FileCheck %s
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
// CHECK-NEXT:     #0 foo1{{.*}} {{.*}}simple_stack2.cc:7{{(:10)?}} (simple_stack2.cc.exe+{{.*}})
// CHECK-NEXT:     #1 bar1{{.*}} {{.*}}simple_stack2.cc:14{{(:3)?}} (simple_stack2.cc.exe+{{.*}})
// CHECK-NEXT:     #2 Thread1{{.*}} {{.*}}simple_stack2.cc:32{{(:3)?}} (simple_stack2.cc.exe+{{.*}})
// CHECK:        Previous read of size 4 at {{.*}} by main thread:
// CHECK-NEXT:     #0 foo2{{.*}} {{.*}}simple_stack2.cc:18{{(:22)?}} (simple_stack2.cc.exe+{{.*}})
// CHECK-NEXT:     #1 bar2{{.*}} {{.*}}simple_stack2.cc:27{{(:3)?}} (simple_stack2.cc.exe+{{.*}})
// CHECK-NEXT:     #2 main{{.*}} {{.*}}simple_stack2.cc:40{{(:3)?}} (simple_stack2.cc.exe+{{.*}})
