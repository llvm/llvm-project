// An access that begins inside a freed chunk and ends past it must be described
// by its first byte, the address in the first line of the report.

// RUN: %clangxx_asan -O0 %s -o %t
// RUN: not %run %t memcpy 2>&1 | FileCheck %s --check-prefixes=CHECK,MEMCPY
// RUN: not %run %t load 2>&1 | FileCheck %s --check-prefixes=CHECK,LOAD

#include <stdlib.h>
#include <string.h>

char dst[128];

int main(int argc, char **argv) {
  if (argc != 2)
    return 1;
  // CHECK: ERROR: AddressSanitizer: heap-use-after-free on address [[ADDR:0x[0-9a-f]+]]
  if (!strcmp(argv[1], "memcpy")) {
    char *p = (char *)malloc(40);
    free(p);
    memcpy(dst, p + 8, 41);
    // MEMCPY: READ of size 41 at [[ADDR]]
    // MEMCPY: [[ADDR]] is located 8 bytes inside of 40-byte region
    return dst[0];
  }
  volatile int *x = (int *)malloc(2 * sizeof(int) + 2);
  free((void *)x);
  return x[2];
  // LOAD: READ of size 4 at [[ADDR]]
  // LOAD: [[ADDR]] is located 8 bytes inside of 10-byte region
  // CHECK: freed by thread T0 here:
}
