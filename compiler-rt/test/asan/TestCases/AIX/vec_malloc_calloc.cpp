// Verify vec_malloc and vec_calloc interceptors

// RUN: %clangxx_asan -O0 %s -o %t
// RUN: not %run %t vec_malloc 2>&1 | FileCheck %s --check-prefix=CHECK-MALLOC
// RUN: not %run %t vec_calloc 2>&1 | FileCheck %s --check-prefix=CHECK-CALLOC

#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
  if (argc != 2)
    return 1;

  char *p;
  if (strcmp(argv[1], "vec_malloc") == 0)
    p = (char *)vec_malloc(10);
  // CHECK-MALLOC: {{READ of size 1 at 0x.* thread T0}}
  // CHECK-MALLOC: {{0x.* is located 0 bytes after 10-byte region}}
  // CHECK-MALLOC: {{0x.* in .vec_malloc}}
  else if (strcmp(argv[1], "vec_calloc") == 0)
    p = (char *)vec_calloc(10, 1);
  // CHECK-CALLOC: {{READ of size 1 at 0x.* thread T0}}
  // CHECK-CALLOC: {{0x.* is located 0 bytes after 10-byte region}}
  // CHECK-CALLOC: {{0x.* in .vec_calloc}}
  else
    return 1;

  char x = p[10];
  free(p);

  return x;
}
