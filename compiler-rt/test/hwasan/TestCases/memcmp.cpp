// RUN: %clangxx_hwasan -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_hwasan -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_hwasan -O2 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_hwasan -O3 %s -o %t && not %run %t 2>&1 | FileCheck %s

#include <string.h>
#include <stdlib.h>
#include <sanitizer/hwasan_interface.h>

int main(int argc, char **argv) {
  __hwasan_enable_allocator_tagging();
  char a[] = {static_cast<char>(argc), 2, 3, 4};
  char *p = (char *)malloc(sizeof(a));
  free(p);
  memcpy(p, a, sizeof(a));
  // CHECK: HWAddressSanitizer: tag-mismatch on address
  return memcmp(p, a, sizeof(a));
}
