// RUN: %clangxx_lowfat -O0 %s -o %t
// RUN: %clangxx_lowfat -O1 %s -o %t
// RUN: %clangxx_lowfat -O2 %s -o %t
// RUN: %clangxx_lowfat -O3 %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

// In-bounds accesses, memcpy, and memset should not report OOB.

#include <cstdio>
#include <cstdlib>
#include <cstring>

int main() {
  char *buf = (char *)malloc(32);
  if (!buf) return 1;

  // Scalar: first and last byte.
  buf[0]  = 'A';
  buf[31] = 'Z';

  // memcpy exactly fits.
  const char src[32] = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
  memcpy(buf, src, 32);

  // memset exactly fits.
  memset(buf, 0, 32);

  // CHECK: OK
  // CHECK-NOT: ERROR: LowFat
  // CHECK-NOT: WARNING: LowFat
  printf("OK\n");

  free(buf);
  return 0;
}
