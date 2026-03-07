// RUN: %clangxx_lowfat -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_lowfat_safe -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s

// OOB write past a 48-byte allocation must be reported.
// This exercises the non-pow2 magic-multiply path.
//
// REQUIRES: lowfat-custom-config

#include <cstdlib>

int main() {
  char *p = (char *)malloc(48);
  if (!p) return 1;

  // Write 8 bytes starting at offset 44.
  // Bytes 44-51 cross the 48-byte allocation boundary.
  // CHECK: LOWFAT ERROR: out-of-bounds error detected!
  double *val = (double *)(p + 44);
  *val = 1.0;

  free(p);
  return 0;
}
