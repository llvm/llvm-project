// RUN: %clangxx_lowfat -fno-builtin-memset -O0 %s -o %t
// RUN: %clangxx_lowfat -fno-builtin-memset -O1 %s -o %t
// RUN: %clangxx_lowfat -fno-builtin-memset -O2 %s -o %t
// RUN: %clangxx_lowfat -fno-builtin-memset -O3 %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s

// Verify that memset writing past the end of a LowFat allocation is detected
// in fatal mode, even when the size isn't a compile-time constant.

#include <string.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  char *buf = (char *)malloc(16);
  if (!buf) return 1;

  // Use argc to prevent the optimizer from knowing the size at compile time.
  // This forces a call to libc memset instead of llvm.memset, which proves
  // our runtime interceptor handles it.
  size_t size = 16 + argc; // 17
  
  // CHECK: LOWFAT ERROR: out-of-bounds error detected!
  // CHECK: operation = write
  // CHECK: size      = 16
  // CHECK: overflow  = +1
  memset(buf, 'A', size);

  free(buf);
  return 0;
}
