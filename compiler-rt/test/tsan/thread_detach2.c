// RUN: %clang_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
#include "test.h"

// Test for https://llvm.org/bugs/show_bug.cgi?id=23235
// The bug was that synchronization between thread creation and thread start
// is not established if pthread_create is followed by pthread_detach.

int x;

void *Thread(void *a) {
  x = 42;
  barrier_wait(&barrier);
  return 0;
}

int main() {
  barrier_init(&barrier, 2);
  pthread_t t;
  x = 43;
  pthread_create(&t, 0, Thread, 0);
  pthread_detach(t);
  barrier_wait(&barrier);
  fprintf(stderr, "PASS\n");
  return 0;
}

// CHECK-NOT: WARNING: ThreadSanitizer: data race
// CHECK: PASS
