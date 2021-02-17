// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
#include "test.h"

int X = 0;

void MySleep() __attribute__((noinline)) {
  sleep(1);  // the sleep that must appear in the report
}

void *Thread(void *p) {
  barrier_wait(&barrier);
  MySleep();  // Assume the main thread has done the write.
  X = 42;
  return 0;
}

int main() {
  barrier_init(&barrier, 2);
  pthread_t t;
  pthread_create(&t, 0, Thread, 0);
  X = 43;
  barrier_wait(&barrier);
  pthread_join(t, 0);
  return 0;
}

// CHECK: WARNING: ThreadSanitizer: data race
// ...
// CHECK:   As if synchronized via sleep:
// CHECK-NEXT:     #0 sleep
// CHECK-NEXT:     #1 MySleep
// CHECK-NEXT:     #2 Thread
