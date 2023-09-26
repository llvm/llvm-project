// RUN: %clangxx_hwasan -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_hwasan -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_hwasan -O2 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_hwasan -O3 %s -o %t && not %run %t 2>&1 | FileCheck %s

// REQUIRES: pointer-tagging

#include <string.h>
int main(int argc, char **argv) {
  char a1[] = {static_cast<char>(argc), 2, 3, 4};
  char a2[] = {1, static_cast<char>(2*argc), 3, 4};
  int res = memcmp(a1, a2, 4 + argc);  // BOOM
  // CHECK: HWAddressSanitizer: tag-mismatch on address
  return res;
}
