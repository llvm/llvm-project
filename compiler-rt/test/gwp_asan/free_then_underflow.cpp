// REQUIRES: gwp_asan
// RUN: %clangxx_gwp_asan %s -o %t
// RUN: %expect_crash %run %t 2>&1 | FileCheck %s

// RUN: %clangxx_gwp_asan %s -o %t -DTOUCH_GUARD_PAGE
// RUN: %expect_crash %run %t 2>&1 | FileCheck %s

// CHECK: GWP-ASan detected a memory error
// CHECK: Use After Free
// CHECK-SAME: warning: buffer overflow/underflow detected on a free()'d allocation
// CHECK-SAME: at 0x{{[a-f0-9]+}} (1 byte to the left

#include <cstdlib>

#include "page_size.h"

int main() {
  unsigned malloc_size = 1;
#ifdef TOUCH_GUARD_PAGE
  malloc_size = pageSize();
#endif // TOUCH_GUARD_PAGE
  char *Ptr = reinterpret_cast<char *>(malloc(malloc_size));
  free(Ptr);
  volatile char x = *(Ptr - 1);
  return 0;
}
