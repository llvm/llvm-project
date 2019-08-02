// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
#include "test.h"

int X = 0;

void *Thread(void *p) {
  X = 42;
  barrier_wait(&barrier);
  return 0;
}

int main() {
  barrier_init(&barrier, 2);
  pthread_t t;
  sleep(1);  // must not appear in the report
  pthread_create(&t, 0, Thread, 0);
  barrier_wait(&barrier);
  X = 43;
  pthread_join(t, 0);
  return 0;
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK-NOT: As if synchronized via sleep
