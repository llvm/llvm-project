// RUN: %clangxx_lowfat %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

// Verify that allocation and free list reuse works correctly, with no OOB errors.

#include <cstdio>

extern "C" void *__lf_malloc(unsigned long size);
extern "C" void __lf_free(void *ptr);

int main() {
  // Allocate and free, then allocate again — should reuse from free list
  int *a = (int *)__lf_malloc(10 * sizeof(int));
  a[0] = 1;
  __lf_free(a);

  int *b = (int *)__lf_malloc(10 * sizeof(int));
  b[0] = 2;
  b[9] = 3;
  __lf_free(b);

  // CHECK: free_list_reuse: ok
  // CHECK-NOT: LOWFAT ERROR
  printf("free_list_reuse: ok\n");
  return 0;
}
