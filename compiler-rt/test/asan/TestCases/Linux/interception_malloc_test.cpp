// ASan interceptor can be accessed with __interceptor_ prefix.

// RUN: %clangxx_asan -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O2 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O3 %s -o %t && not %run %t 2>&1 | FileCheck %s
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

extern "C" void *__interceptor_malloc(size_t size);
extern "C" void *malloc(size_t size) {
  write(2, "malloc call\n", sizeof("malloc call\n") - 1);
  return __interceptor_malloc(size);
}

int main() {
  char *x = (char*)malloc(10 * sizeof(char));
  free(x);
  return (int)strtol(x, 0, 10);
  // CHECK: malloc call
  // CHECK: heap-use-after-free
}
