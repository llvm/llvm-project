// RUN: %clangxx_lowfat_right_align -O0 %s -o %t && %run %t 2>&1 | FileCheck %s

// Documents the known trade-off of right-align mode: underflows into the left
// padding are not caught because the shifted pointer still falls within the
// same 32-byte slot.
//
// 17-byte object in right-align mode: slot = [slot_base, slot_base+32)
//   left padding:  [slot_base,    slot_base+15)  ← blind spot
//   live object:   [slot_base+15, slot_base+32)  ← buf[0]..buf[16]
//
// buf[-1] = slot_base+14, which is inside the slot:
//   GetBase(slot_base+14) = slot_base
//   (slot_base+14 - slot_base) = 14 < 32  → NOT OOB

#include <cstdio>
#include <cstdlib>

int main() {
  // 17 bytes → 32-byte class; object at slot_base+15.
  char *buf = (char *)malloc(17);
  if (!buf) return 1;

  // Write one byte into the left padding (blind spot).
  // This is technically out-of-bounds for the 17-byte allocation, but right-align
  // mode cannot detect it because the access stays within the 32-byte slot.
  buf[-1] = 'X';

  // CHECK: blind spot: not caught (left padding)
  // CHECK-NOT: LOWFAT ERROR
  printf("blind spot: not caught (left padding)\n");
  free(buf);
  return 0;
}
