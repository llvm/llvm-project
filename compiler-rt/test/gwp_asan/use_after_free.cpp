// REQUIRES: gwp_asan
// RUN: %clangxx_gwp_asan %s -o %t
// RUN: %expect_crash %run %t 2>&1 | FileCheck %s

// CHECK: GWP-ASan detected a memory error
// CHECK: Use after free at 0x{{[a-f0-9]+}} (0 bytes into a 10-byte allocation

#include <cstdlib>

int main() {
  char *Ptr = reinterpret_cast<char *>(malloc(10));

  for (unsigned i = 0; i < 10; ++i) {
    *(Ptr + i) = 0x0;
  }

  free(Ptr);
  volatile char x = *Ptr;
  return 0;
}
