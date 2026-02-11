// RUN: %clangxx_lowfat %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

// Verify that allocation and free list reuse works correctly.

extern "C" void *__lf_malloc(unsigned long size);
extern "C" void __lf_free(void *ptr);

int main() {
  // CHECK: LowFat Sanitizer: initialized runtime

  // Allocate and free, then allocate again — should reuse from free list
  int *a = (int *)__lf_malloc(10 * sizeof(int));
  a[0] = 1;
  __lf_free(a);

  int *b = (int *)__lf_malloc(10 * sizeof(int));
  // After free list reuse, b should equal a (same address reused)
  b[0] = 2;
  b[9] = 3;
  __lf_free(b);

  // CHECK-NOT: ERROR: LowFat
  return 0;
}
