// RUN: %clangxx_lowfat_recover %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

// Verify that memset OOB in recover mode warns and continues.

#include <cstdio>
#include <cstdlib>
#include <cstring>

int main() {
  char *dst   = (char *)malloc(16);
  char *guard = (char *)malloc(16);
  if (!dst || !guard) return 1;

  // FIXME: memset with a compile-time constant ensures clang emits the llvm.memset
  // intrinsic, which the LowFat pass instruments. A runtime size would become
  // a plain libc call that the pass cannot see.
  // CHECK: LOWFAT WARNING: out-of-bounds error detected!
  memset(dst, 0, 32); // 32 bytes into a 16-byte allocation → OOB

  // CHECK: after memset
  printf("after memset\n");

  free(guard);
  free(dst);
  return 0;
}
