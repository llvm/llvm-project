// REQUIRES: gwp_asan
// RUN: %clang_gwp_asan %s -g -o %t
// RUN: %expect_crash %t 2>&1 | FileCheck %s

#include <stdlib.h>

__attribute__((noinline)) void *allocate_mem() { return malloc(1); }

__attribute__((noinline)) void free_mem(void *ptr) { free(ptr); }

__attribute__((noinline)) void touch_mem(void *ptr) {
  volatile char sink = *((volatile char *)ptr);
}

// CHECK: Use After Free
// CHECK: touch_mem
// CHECK: was deallocated
// CHECK: free_mem
// CHECK: was allocated
// CHECK: allocate_mem

int main() {
  for (unsigned i = 0; i < 0x10000; ++i) {
    void *ptr = allocate_mem();
    free_mem(ptr);
    touch_mem(ptr);
  }
  return 0;
}
