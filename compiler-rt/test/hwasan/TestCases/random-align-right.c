// Tests malloc_align_right=1 and 8 (randomly aligning right).
// RUN: %clang_hwasan  %s -o %t
//
// RUN: %run %t
// RUN: %env_hwasan_opts=malloc_align_right=1 not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK1
// RUN: %env_hwasan_opts=malloc_align_right=8 not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK8

// REQUIRES: stable-runtime

#include <stdlib.h>
#include <stdio.h>
#include <sanitizer/hwasan_interface.h>

static volatile void *sink;

int main(int argc, char **argv) {
  __hwasan_enable_allocator_tagging();

  // Perform 1000 buffer overflows within the 16-byte granule,
  // so that random right-alignment has a very high chance of
  // catching at least one of them.
  for (int i = 0; i < 1000; i++) {
    char *p = (char*)malloc(20);
    sink = p;
    fprintf(stderr, "[%d] p: %p; accessing p[20]:\n", i, p);
    p[20 * argc] = 0;  // requires malloc_align_right=1 to catch
    fprintf(stderr, "[%d] p: %p; accessing p[30]:\n", i, p);
    p[30 * argc] = 0;  // requires malloc_align_right={1,8} to catch
// CHECK1: accessing p[20]
// CHECK1-NEXT: HWAddressSanitizer: tag-mismatch
// CHECK8: accessing p[30]:
// CHECK8-NEXT: HWAddressSanitizer: tag-mismatch
  }
}

