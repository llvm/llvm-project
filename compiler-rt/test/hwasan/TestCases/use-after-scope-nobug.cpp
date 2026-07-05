// This is the ASAN test of the same name ported to HWAsan.

// RUN: %clangxx_hwasan -O1 %s -o %t && %run %t

// REQUIRES: aarch64-target-arch || riscv64-target-arch

#include <stdio.h>
#include <stdlib.h>

int *p[3];

int main() {
  // Variable goes in and out of scope.
  for (int i = 0; i < 3; i++) {
    int x;
    p[i] = &x;
  }
  printf("PASSED\n");
  return 0;
}
