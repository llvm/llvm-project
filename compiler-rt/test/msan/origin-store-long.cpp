// Check that 8-byte store updates origin for the full store range.
// RUN: %clangxx_msan -fsanitize-memory-track-origins -O0 %s -o %t && not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out && FileCheck %s < %t.out
// RUN: %clangxx_msan -fsanitize-memory-track-origins -O2 %s -o %t && not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out && FileCheck %s < %t.out

#include <sanitizer/msan_interface.h>

int main() {
  uint64_t *volatile p = new uint64_t;
  uint64_t *volatile q = new uint64_t;
  *p = *q;
  char *z = (char *)p;
  return z[6];
// CHECK: WARNING: MemorySanitizer: use-of-uninitialized-value
// CHECK:   in main {{.*}}origin-store-long.cpp:[[@LINE-2]]

// CHECK:  Uninitialized value was created by a heap allocation
// CHECK:   in main {{.*}}origin-store-long.cpp:[[@LINE-8]]
}

