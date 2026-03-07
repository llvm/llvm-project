// RUN: %clangxx_lowfat -O0 %s -o %t && %run %t

// Verify that an in-bounds write to the last byte of a 48-byte allocation
// does not trigger a false positive OOB error.
//
// REQUIRES: lowfat-custom-config

#include <cstdlib>

int main() {
  char *p = (char *)malloc(48);
  if (!p) return 1;

  // Write to the very last byte of the 48-byte allocation.
  // This must NOT trigger a LowFat error.
  p[47] = 'x';

  free(p);
  return 0;
}
