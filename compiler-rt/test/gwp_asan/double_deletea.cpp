// REQUIRES: gwp_asan
// RUN: %clangxx_gwp_asan %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s

// CHECK: GWP-ASan detected a memory error
// CHECK: Double free occurred when trying to free memory at:

#include <cstdlib>

int main() {
  char *Ptr = new char[50];
  delete[] Ptr;
  delete[] Ptr;
  return 0;
}
