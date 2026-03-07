// RUN: %clangxx_lowfat -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_lowfat_safe -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s

// Verify that an OOB write 2 bytes past a 48-byte allocation is detected.
// This test exercises the non-POW2 magic-multiply path: without custom config
// the allocation would be silently rounded to 64 bytes, making p[50] valid.
//
// REQUIRES: lowfat-custom-config

#include <cstdlib>

int main() {
  char *p = (char *)malloc(48);
  if (!p) return 1;

  // Write 8 bytes starting at offset 44.
  // Bytes 44-51 will cross the 48-byte allocation boundary, straddling
  // into the next object grid. This triggers the OOB error.
  // CHECK: LOWFAT ERROR: out-of-bounds error detected!
  double *val = (double *)(p + 44);
  *val = 1.0;

  free(p);
  return 0;
}
