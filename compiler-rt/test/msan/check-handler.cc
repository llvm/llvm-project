// RUN: %clangxx_msan -O0 -g %s -o %t && not %run %t 2>&1 | FileCheck %s

// Verify that CHECK handler prints a stack on CHECK fail.

#include <stdlib.h>

int main(void) {
  // Allocate chunk from the secondary allocator to trigger CHECK(IsALigned())
  // in its free() path.
  void *p = malloc(8 << 20);
  free(reinterpret_cast<char*>(p) + 1);
  // CHECK: MemorySanitizer: bad pointer
  // CHECK: MemorySanitizer CHECK failed
  // CHECK: #0
  return 0;
}
