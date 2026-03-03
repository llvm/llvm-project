// RUN: %clangxx_lowfat_recover %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

// Verify that memcpy OOB in recover mode:
//   1. Prints a WARNING (not ERROR).
//   2. Execution continues past the call (process exits 0).
//   3. The reported overflow is positive (access end past allocation end).

#include <cstdio>
#include <cstdlib>
#include <cstring>

int main() {
  char *dst   = (char *)malloc(16);
  char *guard = (char *)malloc(16);
  if (!dst || !guard) return 1;

  const char payload[32] = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";

  // CHECK: LOWFAT WARNING: out-of-bounds error detected!
  memcpy(dst, payload, 32);

  // Execution must reach here in recover mode.
  // CHECK: after memcpy
  printf("after memcpy\n");

  free(guard);
  free(dst);
  return 0;
}
