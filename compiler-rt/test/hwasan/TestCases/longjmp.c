// RUN: %clang_hwasan -O0 -DNEGATIVE %s -o %t && %run %t 2>&1
// RUN: %clang_hwasan -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s

// REQUIRES: stable-runtime

#include <stdlib.h>
#include <assert.h>
#include <sanitizer/hwasan_interface.h>

__attribute__((noinline))
int f(void *caller_frame) {
  char z[32] = {};
  char *volatile p = z;
  // Tag of local is never zero.
  assert(__hwasan_tag_pointer(p, 0) != p);
#ifndef NEGATIVE
  // This will destroy shadow of "z", and the following load will crash.
  __hwasan_handle_longjmp(caller_frame);
#endif
  return p[0];
}

int main() {
  return f(__builtin_frame_address(0));
  // CHECK: READ of size 8
  // CHECK: pointer tag
  // CHECK: memory tag 0x0
}
