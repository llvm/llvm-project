// RUN: %clangxx_lowfat_right_align -O0 %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_lowfat_right_align -O2 %s -o %t && %run %t 2>&1 | FileCheck %s

// In right-align mode, all accesses within the requested allocation size must
// not trigger OOB (no false positives).
//
// A 48-byte request lands in the 64-byte class. In right-align mode on
// platforms with 16-byte malloc alignment, the object is placed at slot_base+16.
// Bytes buf[0]..buf[47] are all valid.

#include <cstdio>
#include <cstdlib>

int main() {
  // 48 bytes -> 64-byte class; object shifted by an aligned 16-byte offset.
  char *p = (char *)malloc(48);
  if (!p) return 1;

  // Write every byte of the requested allocation.
  for (int i = 0; i < 48; i++)
    p[i] = (char)i;

  // Read back and verify.
  for (int i = 0; i < 48; i++)
    if (p[i] != (char)i) return 2;

  // CHECK: inbounds: ok
  // CHECK-NOT: LOWFAT ERROR
  printf("inbounds: ok\n");
  free(p);
  return 0;
}
