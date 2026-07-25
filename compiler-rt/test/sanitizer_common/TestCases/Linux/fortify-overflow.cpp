// Check that the fortified _chk interceptors keep honoring the destination size
// argument. The write below is in bounds for the allocation, so nothing but the
// fortification bound can detect it.

// RUN: %clangxx -O0 %s -o %t && not --crash %run %t 2>&1 | FileCheck %s
// RUN: %clangxx -O2 %s -o %t && not --crash %run %t 2>&1 | FileCheck %s

// REQUIRES: glibc

#include <stdio.h>
#include <string.h>

extern "C" void *__memset_chk(void *dest, int c, size_t len, size_t destlen);

int main(int argc, char *argv[]) {
  char dest[10];
  // Keep the sizes opaque so that the check is not folded at compile time.
  volatile size_t len = sizeof(dest);
  volatile size_t destlen = sizeof(dest) / 2;

  fprintf(stderr, "before\n");
  // CHECK: before
  __memset_chk(dest, 42, len, destlen);
  fprintf(stderr, "unreachable\n");
  // CHECK-NOT: unreachable
  return 0;
}
