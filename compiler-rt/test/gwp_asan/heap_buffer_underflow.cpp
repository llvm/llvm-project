// REQUIRES: gwp_asan
// RUN: %clangxx_gwp_asan %s -o %t
// RUN: %expect_crash %run %t 2>&1 | FileCheck %s

// CHECK: GWP-ASan detected a memory error
// CHECK: Buffer underflow occurred when accessing memory at:
// CHECK: is located 1 bytes to the left

#include <cstdlib>

#include "page_size.h"

int main() {
  char *Ptr =
      reinterpret_cast<char *>(malloc(pageSize()));
  volatile char x = *(Ptr - 1);
  return 0;
}
