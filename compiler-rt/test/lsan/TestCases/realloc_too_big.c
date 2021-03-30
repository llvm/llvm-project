// RUN: %clang_lsan %s -o %t
// RUN: %env_lsan_opts=allocator_may_return_null=1:max_allocation_size_mb=1:use_stacks=0 not %run %t 2>&1 | FileCheck %s

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

// CHECK: {{Leak|Address}}Sanitizer failed to allocate 0x100001 bytes

// CHECK: {{Leak|Address}}Sanitizer: detected memory leaks
// CHECK: {{Leak|Address}}Sanitizer: 9 byte(s) leaked in 1 allocation(s).

int main() {
  char *p = malloc(9);
  fprintf(stderr, "nine: %p\n", p);
  assert(realloc(p, 0x100001) == NULL); // 1MiB+1
  p = 0;
}
