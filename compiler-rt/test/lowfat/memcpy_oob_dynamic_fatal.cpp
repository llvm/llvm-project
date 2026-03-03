// RUN: %clangxx_lowfat -fno-builtin-memcpy -O0 %s -o %t
// RUN: %clangxx_lowfat -fno-builtin-memcpy -O1 %s -o %t
// RUN: %clangxx_lowfat -fno-builtin-memcpy -O2 %s -o %t
// RUN: %clangxx_lowfat -fno-builtin-memcpy -O3 %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s

// Verify that memcpy writing past the end of a LowFat allocation is detected
// in fatal mode, even when the size isn't a compile-time constant.

#include <string.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  char *dst = (char *)malloc(16);
  char *src = strdup("ABCDEFGHIJKLMNOPQRS"); // 19 bytes
  if (!dst || !src) return 1;

  // Use argc to prevent the optimizer from knowing the size at compile time.
  // This forces a call to libc memcpy instead of llvm.memcpy.
  size_t size = 16 + argc; // 17
  
  // CHECK: LOWFAT ERROR: out-of-bounds error detected!
  // CHECK: operation = write
  // CHECK: size      = 16
  // CHECK: overflow  = +1
  memcpy(dst, src, size);

  free(dst);
  free(src);
  return 0;
}
