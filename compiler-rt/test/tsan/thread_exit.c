// RUN: %clang_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s

// Crashes on powerpc64be
// UNSUPPORTED: powerpc64

#include "test.h"

int var;

void *Thread(void *x) {
  pthread_exit(&var);
  return 0;
}

int main() {
  pthread_t t;
  pthread_create(&t, 0, Thread, 0);
  void *retval = 0;
  pthread_join(t, &retval);
  if (retval != &var) {
    fprintf(stderr, "Unexpected return value\n");
    exit(1);
  }
  fprintf(stderr, "PASS\n");
  return 0;
}

// CHECK-NOT: WARNING: ThreadSanitizer:
// CHECK: PASS
