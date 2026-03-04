// RUN: %clangxx_lowfat -O3 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-ALL
// RUN: %clangxx_lowfat_safe -O3 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-ALL

// Verifies that a genuine OOB heap read is detected in both Fast and Safe mode
// when the loaded value is actually used (returned and passed to printf).

#include <cstdio>
#include <cstdlib>

__attribute__((noinline))
double oob_read(char *p) {
  // 8-byte read at offset 14 of a 16-byte allocation.
  // Bytes [14, 22) exceed the slot boundary [0, 16) → genuine OOB.
  return *(double *)(p + 14);
}

int main() {
  char *p = (char *)malloc(16);
  double val = oob_read(p);  // return value kept live → load not DCE'd
  // CHECK-ALL: LOWFAT ERROR: out-of-bounds error detected!
  printf("val=%f\n", val);
  free(p);
  return 0;
}
