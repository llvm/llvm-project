// RUN: %clangxx_hwasan %s -o %t
// RUN: %env_hwasan_opts=halt_on_error=0 not %run %t 2>&1 | FileCheck %s --implicit-check-not=RETURN_FROM_TEST --check-prefixes=CHECK,RECOVER
// RUN: %env_hwasan_opts=halt_on_error=1 not %run %t 2>&1 | FileCheck %s --implicit-check-not=RETURN_FROM_TEST

#include <sanitizer/hwasan_interface.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

__attribute__((no_sanitize("hwaddress"))) void
ForceCallInterceptor(void *p, int c, size_t size) {
  memset(p, c, size) == nullptr;
}

int main(int argc, char **argv) {
  __hwasan_enable_allocator_tagging();
  char a[] = {static_cast<char>(argc), 2, 3, 4};
  int size = sizeof(a);
  char *volatile p = (char *)malloc(size);
  void *volatile p2 = p;
  for (int i = 0; p2 == p; p2 = __hwasan_tag_pointer(p, ++i)) {
  }
  ForceCallInterceptor(p2, 0, size);
  free(p);
  fprintf(stderr, "RETURN_FROM_TEST\n");
  return 0;
  // CHECK: HWAddressSanitizer: tag-mismatch on address
  // CHECK: WRITE of size 4
  // CHECK: #{{[[:digit:]]+}} 0x{{[[:xdigit:]]+}} in main {{.*}}memset-recover.cpp:[[@LINE-28]]
  // RECOVER: RETURN_FROM_TEST
}
