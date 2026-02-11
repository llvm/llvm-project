// RUN: %clangxx_lowfat %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

// Verify that the LowFat runtime initializes and basic in-bounds
// allocations work without errors.

extern "C" void *__lf_malloc(unsigned long size);
extern "C" void __lf_free(void *ptr);

int main() {
  // CHECK: LowFat Sanitizer: initialized runtime
  int *arr = (int *)__lf_malloc(10 * sizeof(int));
  if (!arr)
    return 1;

  // In-bounds accesses — should not trigger OOB
  arr[0] = 42;
  arr[9] = 99;

  __lf_free(arr);
  // CHECK-NOT: ERROR: LowFat
  return 0;
}
