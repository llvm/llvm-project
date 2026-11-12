// RUN: %clang_hwasan -O0 %s -o %t && %run %t 2>&1 | FileCheck %s

// REQUIRES: pointer-tagging
#include <assert.h>
#include <sanitizer/hwasan_interface.h>
#include <stdlib.h>

__attribute__((noinline)) int f(void *caller_frame) {
  int z = 0;
  int *volatile p = &z;
  // Tag of local is never zero.
  assert(__hwasan_tag_pointer(p, 0) != p);
  __hwasan_handle_longjmp(NULL);
  return p[0];
}

int main() {
  return f(__builtin_frame_address(0));
  // CHECK: HWASan is ignoring requested __hwasan_handle_longjmp:
}
