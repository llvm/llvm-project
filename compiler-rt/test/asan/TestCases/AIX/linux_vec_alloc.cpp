// Verify __linux_vec_malloc, __linux_vec_calloc, and __linux_realloc
// interceptors detect out-of-bounds accesses.

// RUN: %clangxx_asan -O0 %s -o %t
// RUN: not %run %t linux_vec_malloc  2>&1 | FileCheck %s --check-prefix=CHECK-MALLOC
// RUN: not %run %t linux_vec_calloc  2>&1 | FileCheck %s --check-prefix=CHECK-CALLOC
// RUN: not %run %t linux_realloc     2>&1 | FileCheck %s --check-prefix=CHECK-REALLOC

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
  if (strcmp(argv[1], "linux_vec_malloc") == 0)
    p = (char *)__linux_vec_malloc(10);
  // CHECK-MALLOC: {{READ of size 1 at 0x.* thread T0}}
  // CHECK-MALLOC: {{0x.* is located 0 bytes after 10-byte region}}
  // CHECK-MALLOC: {{0x.* in .*__linux_vec_malloc}}
  else if (strcmp(argv[1], "linux_vec_calloc") == 0)
    p = (char *)__linux_vec_calloc(10, 1);
  // CHECK-CALLOC: {{READ of size 1 at 0x.* thread T0}}
  // CHECK-CALLOC: {{0x.* is located 0 bytes after 10-byte region}}
  // CHECK-CALLOC: {{0x.* in .*__linux_vec_calloc}}
  else if (strcmp(argv[1], "linux_realloc") == 0) {
    char *orig = (char *)__linux_vec_malloc(5);
    p = (char *)__linux_realloc(orig, 10);
  }
  // CHECK-REALLOC: {{READ of size 1 at 0x.* thread T0}}
  // CHECK-REALLOC: {{0x.* is located 0 bytes after 10-byte region}}
  // CHECK-REALLOC: {{0x.* in .*__linux_realloc}}
  else
    return 1;

  char x = p[10];
  free(p);
  return x;
}
