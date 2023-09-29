// RUN: %clangxx_hwasan -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_hwasan -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_hwasan -O2 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_hwasan -O3 %s -o %t && not %run %t 2>&1 | FileCheck %s

#include <sanitizer/hwasan_interface.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static __attribute__ ((__noinline__))
char *MakeArray(char* a, int size, int* new_size) {
  char *p = (char *)malloc(size);
  *new_size = size;
  memcpy(p, a, size);
  return p;
}

int main(int argc, char **argv) {
  __hwasan_enable_allocator_tagging();
  char a[] = {static_cast<char>(argc), 2, 3, 4};
  int size = 0;
  char *p = MakeArray(a, sizeof(a), &size);
  free(p);
  // CHECK: HWAddressSanitizer: tag-mismatch on address
  // CHECK: MemcmpInterceptorCommon
  // CHECK: Cause: use-after-free
  int res = memcmp(p, a, size);
  return res;
}
