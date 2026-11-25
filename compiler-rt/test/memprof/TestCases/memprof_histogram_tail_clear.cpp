// A non-granule-multiple allocation owns a partial tail histogram counter that
// must be cleared on allocation; otherwise a recycled chunk leaks the previous
// allocation's count into the last bucket.

// RUN: %clangxx_memprof -O0 -mllvm -memprof-histogram -mllvm -memprof-use-callbacks=true %s -o %t
// RUN: %env_memprof_opts=print_text=1:histogram=1:log_path=stdout %run %t 2>&1 | FileCheck %s

#include <stdio.h>
#include <stdlib.h>

// Distinct call sites keep the two allocations as separate (unmerged) MIBs.
__attribute__((noinline)) static char *alloc_first() {
  return (char *)malloc(20);
}
__attribute__((noinline)) static char *alloc_second() {
  return (char *)malloc(20);
}

int main() {
  // Leave a count of 42 in the tail granule (byte 16), then free.
  char *a = alloc_first();
  if (!a)
    return 1;
  for (int i = 0; i < 42; ++i)
    a[16] = 'A';
  free(a);

  // The stale-tail bug only manifests on a recycled chunk (a fresh chunk's
  // shadow is already zero). memprof has no quarantine, so with no intervening
  // same-size-class allocation the allocator returns the just-freed chunk;
  // require the reuse so the test can't pass vacuously. Its bucket must read 0.
  char *b = alloc_second();
  if (b != a)
    return 1;
  for (int i = 0; i < 5; ++i)
    b[0] = 'B';
  for (int i = 0; i < 7; ++i)
    b[8] = 'C';
  free(b);

  printf("Test completed successfully\n");
  return 0;
}

// CHECK: AccessCountHistogram[3]: 5 7 0
// CHECK: Test completed successfully
