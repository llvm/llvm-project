// RUN: %clangxx_lowfat_right_align -O0 %s -o %t && %run %t 2>&1 | FileCheck %s

// Documents the known trade-off of right-align mode: underflows into the left
// padding are not caught because the shifted pointer still falls within the
// same slot.
//
// 112-byte object in a 128-byte slot with 16-byte malloc alignment:
//   left padding:  [slot_base,    slot_base+16)  <- blind spot
//   live object:   [slot_base+16, slot_base+128) <- buf[0]..buf[111]
//
// buf[-1] = slot_base+15, which is inside the slot:
//   GetBase(slot_base+15) = slot_base
//   (slot_base+15 - slot_base) = 15 < 128 -> NOT OOB

#include <cstdio>
#include <cstdlib>

int main() {
  // 112 bytes -> 128-byte class in both POW2 and custom-config mode.
  char *buf = (char *)malloc(112);
  if (!buf) return 1;

  // Write one byte into the left padding (blind spot).
  // This is technically out-of-bounds for the 112-byte allocation, but
  // right-align mode cannot detect it because the access stays within the same slot.
  buf[-1] = 'X';

  // CHECK: blind spot: not caught (left padding)
  // CHECK-NOT: LOWFAT ERROR
  printf("blind spot: not caught (left padding)\n");
  free(buf);
  return 0;
}
