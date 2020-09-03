// REQUIRES: gwp_asan
// RUN: %clangxx_gwp_asan %s -o %t
// RUN: %expect_crash %run %t 2>&1 | FileCheck %s

// CHECK: GWP-ASan detected a memory error
// CHECK: Buffer Underflow at 0x{{[a-f0-9]+}} (1 byte to the left
// CHECK-SAME: of a {{[1-9][0-9]*}}-byte allocation

#include <cstdlib>

#include "page_size.h"

int main() {
  char *Ptr =
      reinterpret_cast<char *>(malloc(pageSize()));
  volatile char x = *(Ptr - 1);
  return 0;
}
