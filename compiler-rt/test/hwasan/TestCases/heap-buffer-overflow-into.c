// RUN: %clang_hwasan  %s -o %t
// RUN: not %run %t 5 10 2>&1 | FileCheck %s --check-prefix=CHECK5
// RUN: not %run %t 7 10 2>&1 | FileCheck %s --check-prefix=CHECK7
// RUN: not %run %t 8 20 2>&1 | FileCheck %s --check-prefix=CHECK8
// RUN: not %run %t 32 20 2>&1 | FileCheck %s --check-prefix=CHECK32

// REQUIRES: stable-runtime

#include <sanitizer/hwasan_interface.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
  __hwasan_enable_allocator_tagging();
  if (argc < 2) {
    fprintf(stderr, "Invalid number of arguments.");
    abort();
  }
  int read_offset = argc < 2 ? 5 : atoi(argv[1]);
  int size = argc < 3 ? 10 : atoi(argv[2]);
  char *volatile x = (char *)malloc(size);
  memset(x + read_offset, 0, 26);
  // CHECK5: Invalid access starting at offset 5
  // CHECK5: is located 5 bytes inside 10-byte region
  // CHECK7: Invalid access starting at offset 3
  // CHECK7: is located 7 bytes inside 10-byte region
  // CHECK8: Invalid access starting at offset 12
  // CHECK8: is located 8 bytes inside 20-byte region
  // CHECK32: is located 12 bytes to the right of 20-byte region
  free(x);
}
