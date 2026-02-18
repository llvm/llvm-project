// Test that calloc, realloc, and posix_memalign are properly intercepted
// through the Darwin malloc zone mechanism.

// RUN: %clang_memprof -O0 %s -o %t
// RUN: %env_memprof_opts=print_text=true:log_path=stderr:print_module_map=1 %run %t 2>%t.out
// RUN: FileCheck %s < %t.out
// Verify that raw addresses resolve to the expected symbol.
// RUN: %python %S/symbolize_raw_stacks.py %t %t.out | FileCheck --check-prefix=SYM %s

// CHECK: Memory allocation stack id
// CHECK: alloc_count

// SYM: main

#include <stdlib.h>
#include <string.h>

int main() {
  // Test calloc interception.
  int *p = (int *)calloc(10, sizeof(int));
  for (int i = 0; i < 10; i++)
    p[i] = i;
  free(p);

  // Test realloc interception.
  char *q = (char *)malloc(10);
  memset(q, 'a', 10);
  q = (char *)realloc(q, 20);
  memset(q, 'b', 20);
  free(q);

  // Test posix_memalign interception.
  void *r;
  int ret = posix_memalign(&r, 64, 128);
  if (ret == 0) {
    memset(r, 0, 128);
    free(r);
  }

  return 0;
}
