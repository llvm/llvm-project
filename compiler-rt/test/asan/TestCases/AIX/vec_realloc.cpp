// Verify realloc correctly detects overflows.

// RUN: %clangxx_asan -O0 %s -o %t
// RUN: not %run %t realloc_vec_malloc  2>&1 | FileCheck %s --check-prefix=CHECK-REALLOC
// RUN: not %run %t multiple_realloc          2>&1 | FileCheck %s --check-prefix=CHECK-MULTIPLE-REALLOC

#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
  if (argc != 2)
    return 1;

  char *p;
  if (strcmp(argv[1], "realloc_vec_malloc") == 0) {
    char *orig = (char *)vec_malloc(5);
    p = (char *)realloc(orig, 10);
  }
  // CHECK-REALLOC: {{READ of size 1 at 0x.* thread T0}}
  // CHECK-REALLOC: {{0x.* is located 0 bytes after 10-byte region}}
  // CHECK-REALLOC: {{0x.* in .*realloc}}
  else if (strcmp(argv[1], "multiple_realloc") == 0) {
    char *orig = (char *)vec_malloc(5);
    char *r1 = (char *)realloc(orig, 7);
    p = (char *)realloc(r1, 10);
  }
  // CHECK-MULTIPLE-REALLOC: {{READ of size 1 at 0x.* thread T0}}
  // CHECK-MULTIPLE-REALLOC: {{0x.* is located 0 bytes after 10-byte region}}
  // CHECK-MULTIPLE-REALLOC: {{0x.* in .*realloc}}
  else
    return 1;

  char x = p[10];
  free(p);
  return x;
}
