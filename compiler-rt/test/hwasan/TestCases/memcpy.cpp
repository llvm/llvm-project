// RUN: %clangxx_hwasan -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_hwasan -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_hwasan -O2 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_hwasan -O3 %s -o %t && not %run %t 2>&1 | FileCheck %s

#include <sanitizer/hwasan_interface.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

__attribute__((no_sanitize("hwaddress"))) void
ForceCallInterceptor(void *p, const void *a, size_t size) {
  memcpy(p, a, size);
}

int main(int argc, char **argv) {
  __hwasan_enable_allocator_tagging();
  char a[] = {static_cast<char>(argc), 2, 3, 4};
  int size = sizeof(a);
  char *volatile p = (char *)malloc(size);
  free(p);
  ForceCallInterceptor(p, a, size);
  return 0;
  // CHECK: HWAddressSanitizer: tag-mismatch on address
  // CHECK: WRITE of size 4
  // CHECK: #{{[[:digit:]]+}} 0x{{[[:xdigit:]]+}} in main {{.*}}memcpy.cpp:[[@LINE-4]]
  // CHECK: Cause: use-after-free
  // CHECK: freed by thread
  // CHECK: #{{[[:digit:]]+}} 0x{{[[:xdigit:]]+}} in main {{.*}}memcpy.cpp:[[@LINE-8]]
  // CHECK: previously allocated by thread
  // CHECK: #{{[[:digit:]]+}} 0x{{[[:xdigit:]]+}} in main {{.*}}memcpy.cpp:[[@LINE-11]]
}
