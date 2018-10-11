// RUN: %clang_hwasan  %s -o %t
// RUN: not %run %t 40 2>&1 | FileCheck %s --check-prefix=CHECK40
// RUN: not %run %t 80 2>&1 | FileCheck %s --check-prefix=CHECK80
// RUN: not %run %t -30 2>&1 | FileCheck %s --check-prefix=CHECKm30
// RUN: not %run %t -30 1000000 2>&1 | FileCheck %s --check-prefix=CHECKMm30
// RUN: not %run %t 1000000 1000000 2>&1 | FileCheck %s --check-prefix=CHECKM

// REQUIRES: stable-runtime

#include <stdlib.h>
#include <stdio.h>
#include <sanitizer/hwasan_interface.h>

int main(int argc, char **argv) {
  __hwasan_enable_allocator_tagging();
  int offset = argc < 2 ? 40 : atoi(argv[1]);
  int size = argc < 3 ? 30 : atoi(argv[2]);
  char * volatile x = (char*)malloc(size);
  x[offset] = 42;
// CHECK40: allocated heap chunk; size: 32 offset: 8
// CHECK40: is located 10 bytes to the right of 30-byte region
//
// CHECK80: allocated heap chunk; size: 32 offset: 16
// CHECK80: is located 50 bytes to the right of 30-byte region
//
// CHECKm30: allocated heap chunk; size: 32 offset: 2
// CHECKm30: is located 30 bytes to the left of 30-byte region
//
// CHECKMm30: is a large allocated heap chunk; size: 1003520 offset: -30
// CHECKMm30: is located 30 bytes to the left of 1000000-byte region
//
// CHECKM: is a large allocated heap chunk; size: 1003520 offset: 1000000
// CHECKM: is located 0 bytes to the right of 1000000-byte region
  free(x);
}
