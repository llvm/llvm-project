// RUN: %clangxx_lowfat -O0 %s -o %t && %run %t

// In-bounds write at the last byte of a 48-byte allocation should not report OOB.
//
// REQUIRES: lowfat-custom-config

#include <cstdlib>

int main() {
  char *p = (char *)malloc(48);
  if (!p) return 1;

  // Write to the last byte. This must not report a LowFat error.
  p[47] = 'x';

  free(p);
  return 0;
}
