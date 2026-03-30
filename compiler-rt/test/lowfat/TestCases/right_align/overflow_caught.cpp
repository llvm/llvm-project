// RUN: %clangxx_lowfat -O0 %s -o %t && %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-MISS
// RUN: %clangxx_lowfat_right_align -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-CATCH

// Mode-difference test: one-past-end overflow on an allocation where aligned
// right-biasing still leaves a non-zero shift within the slot.
//
// Default (left-align): a 112-byte object lives at the start of a 128-byte
// slot; buf[112] falls in the 16-byte right padding -> access is within the
// slot ->
// NOT caught.
//
// Right-align: the same 112-byte object is shifted to slot_base+16 to preserve
// malloc alignment; buf[112] = slot_base+128, which is exactly the slot
// boundary -> OOB -> caught.

#include <cstdio>
#include <cstdlib>

int main() {
  // 112 bytes -> 128-byte class in both POW2 and custom-config mode.
  char *buf = (char *)malloc(112);
  if (!buf) return 1;

  buf[112] = 'X'; // one-past-end write

  // CHECK-MISS: overflow: not caught (in right padding)
  // CHECK-CATCH: LOWFAT ERROR: out-of-bounds error detected!
  printf("overflow: not caught (in right padding)\n");
  free(buf);
  return 0;
}
