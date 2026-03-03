// RUN: %clangxx_lowfat %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s

// Verify that memset writing past the end of a LowFat allocation is detected
// in fatal mode.

#include <cstdlib>
#include <cstring>

int main() {
  char *dst   = (char *)malloc(16);
  char *guard = (char *)malloc(16); // keep adjacent memory mapped
  if (!dst || !guard) return 1;

  // memset of 32 bytes into a 16-byte allocation — overflows by 16 bytes.
  // CHECK: LOWFAT ERROR: out-of-bounds error detected!
  memset(dst, 0, 32);

  free(guard);
  free(dst);
  return 0;
}
