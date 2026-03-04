// RUN: %clangxx_lowfat_safe_recover -O0 %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_lowfat_safe_recover -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_lowfat_safe_recover -O2 %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_lowfat_safe_recover -O3 %s -o %t && %run %t 2>&1 | FileCheck %s

// Verify that memset OOB in recover mode warns and continues.

#include <cstdio>
#include <cstdlib>
#include <cstring>

int main() {
  char *dst   = (char *)malloc(16);
  char *guard = (char *)malloc(16);
  if (!dst || !guard) return 1;

  // memset of 32 bytes into a 16-byte allocation — overflows by 16 bytes.
  // CHECK: LOWFAT WARNING: out-of-bounds error detected!
  memset(dst, 0, 32);

  // CHECK: after memset
  printf("after memset\n");

  free(guard);
  free(dst);
  return 0;
}
