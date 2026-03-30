// RUN: %clangxx_lowfat -O0 %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_lowfat -O2 %s -o %t && %run %t 2>&1 | FileCheck %s

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

int main() {
  // Requesting 17 bytes will result in a 32-byte LowFat allocation.
  char *p = (char *)malloc(17);
  if (!p) return 1;

  // Access within requested bounds.
  p[0] = 'a';
  p[16] = 'b';

  // Access past requested bounds (17), but within allocation padding (32).
  // LowFat enforces allocation-level bounds, so this should pass.
  p[17] = 'c';
  p[31] = 'z';

  printf("Padding access: ok\n");
  // CHECK: Padding access: ok
  // CHECK-NOT: LOWFAT ERROR

  free(p);
  return 0;
}
