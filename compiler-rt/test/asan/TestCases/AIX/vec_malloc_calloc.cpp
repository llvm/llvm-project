// Verify vec_malloc and vec_calloc interceptors, along with their
// XL-compiler-emitted counterparts __linux_vec_malloc, __linux_vec_calloc,
// and __linux_realloc.

// RUN: %clangxx_asan -O0 %s -o %t
// RUN: not %run %t vec_malloc        2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-VEC-MALLOC
// RUN: not %run %t linux_vec_malloc  2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-LINUX-VEC-MALLOC
// RUN: not %run %t vec_calloc        2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-VEC-CALLOC
// RUN: not %run %t linux_vec_calloc  2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-LINUX-VEC-CALLOC
// RUN: not %run %t linux_realloc     2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-LINUX-REALLOC

#include <stdlib.h>
#include <string.h>

extern "C" {
void *__linux_vec_malloc(unsigned long size);
void *__linux_vec_calloc(unsigned long nmemb, unsigned long size);
void *__linux_realloc(void *ptr, unsigned long size);
}

int main(int argc, char **argv) {
  if (argc != 2)
    return 1;

  char *p;
  if (strcmp(argv[1], "vec_malloc") == 0)
    p = (char *)vec_malloc(10);
  else if (strcmp(argv[1], "linux_vec_malloc") == 0)
    p = (char *)__linux_vec_malloc(10);
  else if (strcmp(argv[1], "vec_calloc") == 0)
    p = (char *)vec_calloc(10, 1);
  else if (strcmp(argv[1], "linux_vec_calloc") == 0)
    p = (char *)__linux_vec_calloc(10, 1);
  else if (strcmp(argv[1], "linux_realloc") == 0) {
    char *orig = (char *)__linux_vec_malloc(5);
    p = (char *)__linux_realloc(orig, 10);
  } else
    return 1;

  // Assertions common to every allocator under test above.
  // CHECK: {{READ of size 1 at 0x.* thread T0}}
  // CHECK: {{0x.* is located 0 bytes after 10-byte region}}
  // CHECK: {{allocated by thread T0 here:}}
  //
  // The allocation frame (not just some caller further up the backtrace)
  // must be the specific symbol under test, otherwise this only proves that
  // the real libc symbol forwarded to some other, already-intercepted
  // function.
  // CHECK-VEC-MALLOC-NEXT: {{#0 .* in .vec_malloc}}
  // CHECK-LINUX-VEC-MALLOC-NEXT: {{#0 .* in .*__linux_vec_malloc}}
  // CHECK-VEC-CALLOC-NEXT: {{#0 .* in .vec_calloc}}
  // CHECK-LINUX-VEC-CALLOC-NEXT: {{#0 .* in .*__linux_vec_calloc}}
  // CHECK-LINUX-REALLOC-NEXT: {{#0 .* in .*__linux_realloc}}
  char x = p[10];
  free(p);

  return x;
}
