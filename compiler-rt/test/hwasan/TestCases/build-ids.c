// RUN: %clang_hwasan -Wl,--build-id=0xaba493998257fbdd %s -o %t
// RUN: %env_hwasan_opts=symbolize=0 not %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK,NOSYM
// RUN: not %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK,SYM

#include <stdlib.h>

#include <sanitizer/hwasan_interface.h>

int main(int argc, char **argv) {
  __hwasan_enable_allocator_tagging();
  char *buf = (char *)malloc(1);
  buf[32] = 'x';
  // CHECK: ERROR: HWAddressSanitizer: tag-mismatch
  // NOSYM:  0x{{.*}}  {{.*}}build-ids.c{{.*}} (BuildId: aba493998257fbdd)
  // SYM:  0x{{.*}} in main {{.*}}build-ids.c:[[@LINE-3]]:{{[0-9]+}}
  return 0;
}
