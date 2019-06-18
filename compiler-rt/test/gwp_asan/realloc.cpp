// REQUIRES: gwp_asan
// RUN: %clangxx_gwp_asan %s -o %t -DTEST_MALLOC
// RUN: not %run %t 2>&1 | FileCheck %s --check-prefix CHECK-MALLOC

// Check both C++98 and C.
// RUN: %clangxx_gwp_asan -std=c++98 %s -o %t -DTEST_FREE
// RUN: %expect_crash %run %t 2>&1 | FileCheck %s --check-prefix CHECK-FREE
// RUN: cp %s %t.c && %clang_gwp_asan %t.c -o %t -DTEST_FREE
// RUN: %expect_crash %run %t 2>&1 | FileCheck %s --check-prefix CHECK-FREE

// Ensure GWP-ASan stub implementation of realloc() in Scudo works to-spec. In
// particular, the behaviour regarding realloc of size zero is interesting, as
// it's defined as free().

#include <stdlib.h>

int main() {
#if defined(TEST_MALLOC)
  // realloc(nullptr, size) is equivalent to malloc(size).
  char *Ptr = reinterpret_cast<char *>(realloc(nullptr, 1));
  *Ptr = 0;
  // Trigger an INVALID_FREE to the right.
  free(Ptr + 1);

  // CHECK-MALLOC: GWP-ASan detected a memory error
  // CHECK-MALLOC: Invalid (wild) free occurred when trying to free memory at:
  // CHECK-MALLOC: is located 1 bytes to the right of a 1-byte allocation
#elif defined(TEST_FREE)
  char *Ptr = (char *) malloc(1);
  // realloc(ptr, 0) is equivalent to free(ptr) and must return nullptr. Note
  // that this is only the specification in C++98 and C.
  if (realloc(Ptr, 0) != NULL) {

  }
  // Trigger a USE_AFTER_FREE.
  *Ptr = 0;

  // CHECK-FREE: GWP-ASan detected a memory error
  // CHECK-FREE: Use after free occurred when accessing memory at:
  // CHECK-FREE: is a 1-byte allocation
#endif

  return 0;
}
