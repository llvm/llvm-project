// RUN: %clangxx_hwasan %s -o %t && not %run %t 2>&1 | FileCheck %s

#include <stdio.h>
#include <stdlib.h>

#include <sanitizer/hwasan_interface.h>

#include "utils.h"

__attribute__((no_sanitize("hwaddress"))) extern "C" void callback(const char *msg) {
  untag_fprintf(stderr, "== error start\n%s\n== error end\n", msg);
}

int main() {
  __hwasan_enable_allocator_tagging();
  __hwasan_set_error_report_callback(&callback);
  char *volatile p = (char *)malloc(16);
  p[16] = 1;
  free(p);
  // CHECK: ERROR: HWAddressSanitizer:
  // CHECK: WRITE of size 1 at
  // CHECK: allocated here:
  // CHECK: Memory tags around the buggy address

  // CHECK: == error start
  // CHECK: ERROR: HWAddressSanitizer:
  // CHECK: WRITE of size 1 at
  // CHECK: allocated here:
  // CHECK: Memory tags around the buggy address
  // CHECK: == error end
}
