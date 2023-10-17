// RUN: %clangxx_hwasan -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_hwasan -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_hwasan -O2 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_hwasan -O3 %s -o %t && not %run %t 2>&1 | FileCheck %s

#include <sanitizer/hwasan_interface.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char **argv) {
  __hwasan_enable_allocator_tagging();
  char a[] = {static_cast<char>(argc), 2, 3, 4};
  int size = sizeof(a);
  char *p = (char *)malloc(size);
  memcpy(p, a, size);
  free(p);
  return bcmp(p, a, size);
  // CHECK: HWAddressSanitizer: tag-mismatch on address
  // CHECK: READ of size 4
  // CHECK: #{{[[:digit:]]+}} 0x{{[[:xdigit:]]+}} in main {{.*}}bcmp.cpp:[[@LINE-3]]
  // CHECK: Cause: use-after-free
  // CHECK: freed by thread
  // CHECK: #{{[[:digit:]]+}} 0x{{[[:xdigit:]]+}} in main {{.*}}bcmp.cpp:[[@LINE-7]]
  // CHECK: previously allocated by thread
  // CHECK: #{{[[:digit:]]+}} 0x{{[[:xdigit:]]+}} in main {{.*}}bcmp.cpp:[[@LINE-11]]
}
