// REQUIRES: gwp_asan
// RUN: %clangxx_gwp_asan %s -o %t
// RUN: %expect_crash %run %t 2>&1 | FileCheck %s

#include <cstdlib>

int main() {
  // CHECK: GWP-ASan detected a memory error
  // CHECK: Double Free at 0x{{[a-f0-9]+}} (a 10-byte allocation)
  void *Ptr = malloc(10);

  free(Ptr);
  free(Ptr);
  return 0;
}
