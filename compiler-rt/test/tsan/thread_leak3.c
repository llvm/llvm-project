// RUN: %clang_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
#include "test.h"

void *Thread(void *x) {
  barrier_wait(&barrier);
  return 0;
}

int main() {
  barrier_init(&barrier, 2);
  pthread_t t;
  pthread_create(&t, 0, Thread, 0);
  barrier_wait(&barrier);
  sleep(1);  // wait for the thread to finish and exit
  return 0;
}

// CHECK: WARNING: ThreadSanitizer: thread leak
// CHECK: SUMMARY: ThreadSanitizer: thread leak{{.*}}main
