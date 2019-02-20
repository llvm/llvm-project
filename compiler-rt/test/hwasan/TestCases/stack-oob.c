// RUN: %clang_hwasan -DSIZE=16 -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clang_hwasan -DSIZE=64 -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clang_hwasan -DSIZE=0x1000 -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s

// REQUIRES: stable-runtime

#include <stdlib.h>
#include <sanitizer/hwasan_interface.h>

__attribute__((noinline))
int f() {
  char z[SIZE];
  char *volatile p = z;
  return p[SIZE];
}

int main() {
  return f();
  // CHECK: READ of size 1 at
  // CHECK: #0 {{.*}} in f{{.*}}stack-oob.c:14

  // CHECK: is located in stack of threa

  // CHECK: SUMMARY: HWAddressSanitizer: tag-mismatch {{.*}} in f
}
