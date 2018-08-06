// RUN: %clangxx_hwasan -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s

// REQUIRES: stable-runtime

#include <stdlib.h>
#include <sanitizer/hwasan_interface.h>

__attribute__((noinline))
char *f() {
  char z[0x1000];
  char *volatile p = z;
  return p;
}

int main() {
  return *f();
  // CHECK: READ of size 1 at
  // CHECK: #0 {{.*}} in main{{.*}}stack-uar.cc:16

  // CHECK: HWAddressSanitizer can not describe address in more detail.

  // CHECK: SUMMARY: HWAddressSanitizer: tag-mismatch {{.*}} in main
}
