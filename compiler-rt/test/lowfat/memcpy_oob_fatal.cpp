// RUN: %clangxx_lowfat %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s

// Verify that memcpy writing past the end of a LowFat allocation is detected
// in fatal mode.  The process must exit with a non-zero code (checked by
// "not %run") and print the expected diagnostic.

#include <cstdlib>
#include <cstring>

int main() {
  char *dst   = (char *)malloc(16);
  char *guard = (char *)malloc(16); // keep adjacent memory mapped
  if (!dst || !guard) return 1;

  const char payload[32] = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";

  // memcpy of 32 bytes into a 16-byte allocation — overflows by 16 bytes.
  // CHECK: LOWFAT ERROR: out-of-bounds error detected!
  memcpy(dst, payload, 32);

  free(guard);
  free(dst);
  return 0;
}
