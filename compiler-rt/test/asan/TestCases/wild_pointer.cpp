// RUN: %clangxx_asan %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s
// REQUIRES: asan-64-bits

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

int main() {
  char *p = new char;
  char *dest = new char;
  const size_t offset = 0x4567890123456789;
  // Flush it so the output came out before the asan report.
  fprintf(stderr, "Expected bad addr: %p\n", p + offset);
  fflush(stderr);
  memmove(dest, p, offset);
  return 0;
}

// CHECK: Expected bad addr: [[ADDR:0x[0-9,a-f]+]]
// CHECK: AddressSanitizer: unknown-crash on address [[ADDR]]
// CHECK: Address [[ADDR]] is a wild pointer inside of access range of size 0x4567890123456789
