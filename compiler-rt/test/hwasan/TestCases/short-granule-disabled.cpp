// RUN: %clangxx_hwasan %s -o %t && %run %t 2>&1

#include <sanitizer/hwasan_interface.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Regression test for https://reviews.llvm.org/D107938#2961070, where, on
// reusing an allocation, we forgot to reset the short granule tag if the
// allocator was disabled. This lead to a false positive magic-string mismatch.

int main() {
  void *p = malloc(16);
  memset(p, 0xff, 16);
  free(p);

  // Relies on the LRU cache immediately recycling the allocation above.
  p = malloc(8);
  free(p); // Regression was here, in the magic-string check in the runtime.
  return 0;
}
