// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
// UNSUPPORTED: powerpc64le
#include <stdio.h>

// Defined by tsan.
extern "C" void *__interceptor_malloc(unsigned long size);
extern "C" void __interceptor_free(void *p);

extern "C" void *malloc(unsigned long size) {
  static int first = 0;
  if (__sync_lock_test_and_set(&first, 1) == 0)
    fprintf(stderr, "user malloc\n");
  return __interceptor_malloc(size);
}

extern "C" void free(void *p) {
  __interceptor_free(p);
}

int main() {
  volatile char *p = (char*)malloc(10);
  p[0] = 0;
  free((void*)p);
}

// CHECK: user malloc
// CHECK-NOT: ThreadSanitizer

