// RUN: %clangxx_asan %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s
// REQUIRES: asan-64-bits

#include <inttypes.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

int main() {
  char *p = new char;
  char *dest = new char;
  const size_t size = 0x4567890123456789;

  // The output here needs to match the output from the sanitizer runtime,
  // which includes 0x and prints hex in lower case.
  //
  // On Windows, %p omits %0x and prints hex characters in upper case,
  // so we use PRIxPTR instead of %p.
  fprintf(stderr, "Expected bad addr: %#" PRIxPTR "\n",
          reinterpret_cast<uintptr_t>(p));
  // Flush it so the output came out before the asan report.
  fflush(stderr);

  memmove(dest, p, size);
  return 0;
}

// CHECK: Expected bad addr: [[ADDR:0x[0-9,a-f]+]]
// CHECK: AddressSanitizer: heap-buffer-overflow on address [[ADDR]]
// CHECK: READ of size 5001116549197948809 at [[ADDR]] thread T0
