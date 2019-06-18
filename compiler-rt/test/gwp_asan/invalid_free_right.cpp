// REQUIRES: gwp_asan
// RUN: %clangxx_gwp_asan %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s

// CHECK: GWP-ASan detected a memory error
// CHECK: Invalid (wild) free occurred when trying to free memory at:
// CHECK: is located 1 bytes to the right

#include <cstdlib>

int main() {
  char *Ptr =
      reinterpret_cast<char *>(malloc(1));
  free(Ptr + 1);
  return 0;
}
