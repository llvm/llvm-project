// RUN: %clangxx_lowfat -O0 %s -o %t
// RUN: %clangxx_lowfat -O1 %s -o %t
// RUN: %clangxx_lowfat -O2 %s -o %t
// RUN: %clangxx_lowfat -O3 %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s

// Verify that an OOB load (scalar read across allocation boundary) is detected
// in fatal mode.

#include <cstdlib>

int main() {
  char *buf = (char *)malloc(32);
  if (!buf) return 1;

  buf[0] = 'H';
  buf[31] = 'i';

  // Read 8 bytes at offset 28 of a 32-byte allocation:
  //   bytes 28–35 exceed the 32-byte boundary → OOB.
  // CHECK: LOWFAT ERROR: out-of-bounds error detected!
  // FIXME: must NOT use volatile here — the LowFat pass skips volatile accesses.
  double *p = (double *)(buf + 28);
  double val = *p; // 8-byte read at offset 28 of 32-byte alloc → OOB (bytes 28–35)
  (void)val;       // keep live to prevent DSE; crash fires on the load above

  free(buf);
  return 0;
}
