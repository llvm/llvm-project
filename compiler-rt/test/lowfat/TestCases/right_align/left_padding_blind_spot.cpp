// RUN: %clangxx_lowfat_right_align -O0 %s -o %t && %run %t 2>&1 | FileCheck %s

// Documents the known trade-off of right-align mode: underflows into the left
// padding are not caught because the shifted pointer still falls within the
// same slot.
//
// 48-byte object in a 64-byte slot with 16-byte malloc alignment:
//   left padding:  [slot_base,    slot_base+16)  <- blind spot
//   live object:   [slot_base+16, slot_base+64)  <- buf[0]..buf[47]
//
// buf[-1] = slot_base+15, which is inside the slot:
//   GetBase(slot_base+15) = slot_base
//   (slot_base+15 - slot_base) = 15 < 64  -> NOT OOB

#include <cstdio>
#include <cstdlib>

int main() {
  // 48 bytes -> 64-byte class; object at slot_base+16.
  char *buf = (char *)malloc(48);
  if (!buf) return 1;

  // Write one byte into the left padding (blind spot).
  // This is technically out-of-bounds for the 48-byte allocation, but
  // right-align mode cannot detect it because the access stays within the same slot.
  buf[-1] = 'X';

  // CHECK: blind spot: not caught (left padding)
  // CHECK-NOT: LOWFAT ERROR
  printf("blind spot: not caught (left padding)\n");
  free(buf);
  return 0;
}
