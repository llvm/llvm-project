// RUN: %clangxx_lowfat -O0 %s -o %t && %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-MISS
// RUN: %clangxx_lowfat_right_align -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-CATCH

// Mode-difference test: one-past-end overflow on an allocation where aligned
// right-biasing still leaves a non-zero shift within the slot.
//
// Default (left-align): a 48-byte object lives at the start of a 64-byte slot;
// buf[48] falls in the 16-byte right padding -> access is within the slot ->
// NOT caught.
//
// Right-align: the same 48-byte object is shifted to slot_base+16 to preserve
// malloc alignment; buf[48] = slot_base+64, which is exactly the slot boundary
// -> OOB -> caught.

#include <cstdio>
#include <cstdlib>

int main() {
  // 48 bytes -> 64-byte class.
  char *buf = (char *)malloc(48);
  if (!buf) return 1;

  buf[48] = 'X'; // one-past-end write

  // CHECK-MISS: overflow: not caught (in right padding)
  // CHECK-CATCH: LOWFAT ERROR: out-of-bounds error detected!
  printf("overflow: not caught (in right padding)\n");
  free(buf);
  return 0;
}
