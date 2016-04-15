// RUN: %clang_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
#include "test.h"

void *Thread(void *x) {
  sleep(100);  // leave the thread "running"
  return 0;
}

int main() {
  pthread_t t;
  pthread_create(&t, 0, Thread, 0);
  fprintf(stderr, "DONE\n");
  return 0;
}

// CHECK: DONE
// CHECK-NOT: WARNING: ThreadSanitizer: thread leak

