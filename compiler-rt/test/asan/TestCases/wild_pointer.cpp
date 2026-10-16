// RUN: %clangxx_asan %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s
// REQUIRES: asan-64-bits

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

int main() {
  char *dest = new char;
  const char *src =
      reinterpret_cast<const char *>(static_cast<uintptr_t>(0x1234));
  const size_t size = 0x4567890123456789ULL;

  // The output here needs to match the output from the sanitizer runtime,
  // which includes 0x and prints hex in lower case.
  //
  // On Windows, %p omits %0x and prints hex characters in upper case,
  // so we use PRIxPTR instead of %p.
  fprintf(stderr, "Expected bad addr: %#" PRIxPTR "\n",
          reinterpret_cast<uintptr_t>(src));
  // Flush it so the output came out before the asan report.
  fflush(stderr);

  memmove(dest, src, size);
  return 0;
}

// CHECK: Expected bad addr: 0x[[ADDR:[0-9a-f]+]]
// CHECK: AddressSanitizer: unknown-crash on address {{0x0*}}[[ADDR]]
// CHECK: READ of size 5001116549197948809 at {{0x0*}}[[ADDR]] thread T0
// CHECK: Address {{0x0*}}[[ADDR]] is a wild pointer inside of access range of size 0x4567890123456789
