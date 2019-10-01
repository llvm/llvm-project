// Tests malloc_align_right=1 and 8 (randomly aligning right).
// RUN: %clang_hwasan  %s -o %t
//
// RUN: %run %t 20
// RUN: %run %t 30
// RUN: %env_hwasan_opts=malloc_align_right=1 not %run %t 20 2>&1 | FileCheck %s --check-prefix=CHECK20
// RUN: %env_hwasan_opts=malloc_align_right=1 not %run %t 30 2>&1 | FileCheck %s --check-prefix=CHECK30
// RUN: %env_hwasan_opts=malloc_align_right=8 not %run %t 30 2>&1 | FileCheck %s --check-prefix=CHECK30

// REQUIRES: stable-runtime

#include <stdlib.h>
#include <stdio.h>
#include <sanitizer/hwasan_interface.h>

static volatile void *sink;

int main(int argc, char **argv) {
  __hwasan_enable_allocator_tagging();
  int index = atoi(argv[1]);

  // Perform 1000 buffer overflows within the 16-byte granule,
  // so that random right-alignment has a very high chance of
  // catching at least one of them.
  for (int i = 0; i < 1000; i++) {
    char *p = (char*)malloc(20);
    sink = p;
    p[index] = 0;
// index=20 requires malloc_align_right=1 to catch
// CHECK20: HWAddressSanitizer: tag-mismatch
// index=30 requires malloc_align_right={1,8} to catch
// CHECK30: HWAddressSanitizer: tag-mismatch
  }
}

