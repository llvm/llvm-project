// RUN: %clangxx_tsan -O1 %s -o %t
// RUN: %env_tsan_opts=allocator_may_return_null=1 %run %t 2>&1 | FileCheck %s
#include <stdio.h>
#include <stdlib.h>

int main() {
  void *p = malloc((size_t)-1);
  if (p != 0)
    fprintf(stderr, "FAIL malloc(-1) = %p\n", p);
  p = malloc((size_t)-1 / 2);
  if (p != 0)
    fprintf(stderr, "FAIL malloc(-1/2) = %p\n", p);
  p = calloc((size_t)-1, (size_t)-1);
  if (p != 0)
    fprintf(stderr, "FAIL calloc(-1, -1) = %p\n", p);
  p = calloc((size_t)-1 / 2, (size_t)-1 / 2);
  if (p != 0)
    fprintf(stderr, "FAIL calloc(-1/2, -1/2) = %p\n", p);
  fprintf(stderr, "OK\n");
}

// CHECK-NOT: FAIL
// CHECK-NOT: failed to allocate
