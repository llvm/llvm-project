// RUN: %clangxx_lowfat -O0 %s -o %t && %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-MISS
// RUN: %clangxx_lowfat_right_align -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-CATCH

// Mode-difference test: one-past-end overflow on a non-POW2-sized allocation.
//
// Default (left-align): 17-byte object at slot_base; buf[17] falls in the
// 15-byte right padding → access is within the 32-byte slot → NOT caught.
//
// Right-align: 17-byte object at slot_base+15; buf[17] = slot_base+32, which
// is exactly the slot boundary → OOB → caught.

#include <cstdio>
#include <cstdlib>

int main() {
  // 17 bytes → 32-byte class.
  char *buf = (char *)malloc(17);
  if (!buf) return 1;

  buf[17] = 'X'; // one-past-end write

  // CHECK-MISS: overflow: not caught (in right padding)
  // CHECK-CATCH: LOWFAT ERROR: out-of-bounds error detected!
  printf("overflow: not caught (in right padding)\n");
  free(buf);
  return 0;
}
