// RUN: %clangxx_hwasan -DERR=1 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_hwasan -DERR=2 %s -o %t && not %run %t 2>&1 | FileCheck %s
// REQUIRES: android

#include <stdlib.h>
#include <stdio.h>

#include <sanitizer/hwasan_interface.h>

__attribute__((no_sanitize("hwaddress")))
extern "C" void android_set_abort_message(const char *msg) {
  fprintf(stderr, "== abort message start\n%s\n== abort message end\n", msg);
}

int main() {
  __hwasan_enable_allocator_tagging();
  char *volatile p = (char *)malloc(16);
  if (ERR==1) {
    p[16] = 1;
  } else {
    free(p);
    free(p);
  }
  // CHECK: ERROR: HWAddressSanitizer:
  // CHECK: == abort message start
  // CHECK: ERROR: HWAddressSanitizer:
  // CHECK: == abort message end
}
