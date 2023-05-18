// REQUIRES: gwp_asan
// RUN: %clangxx_gwp_asan %s -o %t
// RUN: %expect_crash %run %t 2>&1 | FileCheck %s

// CHECK: GWP-ASan detected a memory error
// CHECK: Double Free at 0x{{[a-f0-9]+}} (a 50-byte allocation)

#include <cstdlib>

int main() {
  char *Ptr = new char[50];
  delete[] Ptr;
  delete[] Ptr;
  return 0;
}
