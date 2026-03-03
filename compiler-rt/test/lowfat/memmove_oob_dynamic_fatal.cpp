// RUN: %clangxx_lowfat -fno-builtin-memmove -O0 %s -o %t
// RUN: %clangxx_lowfat -fno-builtin-memmove -O1 %s -o %t
// RUN: %clangxx_lowfat -fno-builtin-memmove -O2 %s -o %t
// RUN: %clangxx_lowfat -fno-builtin-memmove -O3 %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s

// Verify that memmove writing past the end of a LowFat allocation is detected

#include <string.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  char *dst = (char *)malloc(16);
  char *src = dst + 4;
  if (!dst) return 1;

  // Use argc to prevent the optimizer from knowing the size at compile time.
  size_t size = 12 + argc; // 13. dst+13 is within 16, but src+13 = dst+17, which is OOB (+1)
  
  // CHECK: LOWFAT ERROR: out-of-bounds error detected!
  // CHECK: operation = read
  // CHECK: size      = 16
  // CHECK: overflow  = +1
  memmove(dst, src, size);

  free(dst);
  return 0;
}
