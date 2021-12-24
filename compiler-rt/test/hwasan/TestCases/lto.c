// Test globals with LTO, since it invokes the integrated assembler separately.
// RUN: %clang_hwasan -flto %s -o %t
// RUN: not %run %t 1 2>&1 | FileCheck %s

// REQUIRES: pointer-tagging, x86_64-target-arch

#include <stdlib.h>

int x = 1;

int main(int argc, char **argv) {
  // CHECK: Cause: global-overflow
  // CHECK: is located 0 bytes to the right of 4-byte global variable x {{.*}} in {{.*}}lto.c.tmp
  // CHECK-NOT: can not describe
  (&x)[atoi(argv[1])] = 1;
}
