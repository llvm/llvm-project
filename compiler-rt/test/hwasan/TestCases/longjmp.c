// RUN: %clang_hwasan -O0 -DNEGATIVE %s -o %t && %run %t 2>&1
// RUN: %clang_hwasan -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s

// REQUIRES: stable-runtime, pointer-tagging

#include <setjmp.h>
#include <stdlib.h>
#include <assert.h>
#include <sanitizer/hwasan_interface.h>
#include <unistd.h>

static int *volatile p;

__attribute__((noinline))
int f(jmp_buf buf) {
  int z = 0;
  p = &z;
  // Tag of local is never zero.
  assert(__hwasan_tag_pointer(p, 0) != p);
#ifndef NEGATIVE
  // This will destroy shadow of "z", the p[0] in main will crash.
  longjmp(buf, 1);
#endif
  return p[0];
}

int main() {
  jmp_buf buf;
  if (setjmp(buf)) {
    return p[0];
  } else {
    f(buf);
  }
  return 0;
  // CHECK: READ of size 4 at {{.*}} tags: {{.*}}/00 (ptr/mem)
}
