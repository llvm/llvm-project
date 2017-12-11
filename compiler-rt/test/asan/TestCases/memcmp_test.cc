// RUN: %clangxx_asan -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O2 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O3 %s -o %t && not %run %t 2>&1 | FileCheck %s

// REQUIRES: compiler-rt-optimized

#include <string.h>
int main(int argc, char **argv) {
  char a1[] = {static_cast<char>(argc), 2, 3, 4};
  char a2[] = {1, static_cast<char>(2*argc), 3, 4};
  int res = memcmp(a1, a2, 4 + argc);  // BOOM
  // CHECK: AddressSanitizer: stack-buffer-overflow
  // CHECK: {{#0.*memcmp}}
  // CHECK: {{#1.*main}}
  return res;
}
