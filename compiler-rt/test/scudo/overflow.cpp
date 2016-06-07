// RUN: %clang_scudo %s -o %t
// RUN:                                  not %run %t malloc     2>&1 | FileCheck %s
// RUN: SCUDO_OPTIONS=QuarantineSizeMb=1 not %run %t quarantine 2>&1 | FileCheck %s

// Tests that header corruption of an allocated or quarantined chunk is caught.

#include <assert.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
  assert(argc == 2);
  if (!strcmp(argv[1], "malloc")) {
    // Simulate a header corruption of an allocated chunk (1-bit)
    void *p = malloc(1U << 4);
    if (!p)
      return 1;
    ((char *)p)[-1] ^= 1;
    free(p);
  }
  if (!strcmp(argv[1], "quarantine")) {
    void *p = malloc(1U << 4);
    if (!p)
      return 1;
    free(p);
    // Simulate a header corruption of a quarantined chunk
    ((char *)p)[-2] ^= 1;
    // Trigger the quarantine recycle
    for (int i = 0; i < 0x100; i++) {
      p = malloc(1U << 16);
      free(p);
    }
  }
  return 0;
}

// CHECK: ERROR: corrupted chunk header at address
