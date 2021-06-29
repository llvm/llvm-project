// RUN: %clang_hwasan  %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK

// REQUIRES: stable-runtime

#include <sanitizer/hwasan_interface.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
  __hwasan_enable_allocator_tagging();
  char *volatile x = (char *)malloc(10);
  memset(x + 5, 0, 26);
  // CHECK: is located 5 bytes inside 10-byte region
  free(x);
}
