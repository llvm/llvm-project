// RUN: %clangxx_lowfat_right_align -O0 %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_lowfat_right_align -O2 %s -o %t && %run %t 2>&1 | FileCheck %s

// In right-align mode, all accesses within the requested allocation size must
// not trigger OOB (no false positives).
//
// A 17-byte request lands in the 32-byte class. In right-align mode the object
// is placed at slot_base+15, so its right edge coincides with the slot boundary
// at slot_base+32. Bytes buf[0]..buf[16] are all valid.

#include <cstdio>
#include <cstdlib>

int main() {
  // 17 bytes → 32-byte class; object at slot_base+15 in right-align mode.
  char *p = (char *)malloc(17);
  if (!p) return 1;

  // Write every byte of the requested allocation.
  for (int i = 0; i < 17; i++)
    p[i] = (char)i;

  // Read back and verify.
  for (int i = 0; i < 17; i++)
    if (p[i] != (char)i) return 2;

  // CHECK: inbounds: ok
  // CHECK-NOT: LOWFAT ERROR
  printf("inbounds: ok\n");
  free(p);
  return 0;
}
