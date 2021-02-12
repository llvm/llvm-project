// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include "test.h"

void *Thread1(void *p) {
  *(int*)p = 42;
  return 0;
}

void *Thread2(void *p) {
  *(int*)p = 44;
  return 0;
}

void *alloc() __attribute__((noinline)) {
  return malloc(99);
}

void *AllocThread(void* arg) {
  return alloc();
}

int main() {
  void *p = 0;
  pthread_t t[2];
  pthread_create(&t[0], 0, AllocThread, 0);
  pthread_join(t[0], &p);
  print_address("addr=", 1, p);
  pthread_create(&t[0], 0, Thread1, (char*)p + 16);
  pthread_create(&t[1], 0, Thread2, (char*)p + 16);
  pthread_join(t[0], 0);
  pthread_join(t[1], 0);
  return 0;
}

// CHECK: addr=[[ADDR:0x[0-9,a-f]+]]
// CHECK: WARNING: ThreadSanitizer: data race
// ...
// CHECK: Location is heap block of size 99 at [[ADDR]] allocated by thread T1:
// CHECK:     #0 malloc
// CHECK:     #{{1|2}} alloc
// CHECK:     #{{2|3}} AllocThread
// ...
// CHECK:   Thread T1 (tid={{.*}}, finished) created by main thread at:
// CHECK:     #0 pthread_create
// CHECK:     #1 main
