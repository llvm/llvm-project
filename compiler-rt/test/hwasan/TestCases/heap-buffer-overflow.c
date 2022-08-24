// RUN: %clang_hwasan  %s -o %t
// RUN: not %run %t 40 2>&1 | FileCheck %s --check-prefix=CHECK40
// RUN: not %run %t 80 2>&1 | FileCheck %s --check-prefix=CHECK80
// RUN: not %run %t -30 2>&1 | FileCheck %s --check-prefix=CHECKm30
// RUN: not %run %t -30 1000000 2>&1 | FileCheck %s --check-prefix=CHECKMm30
// RUN: not %run %t 1000000 1000000 2>&1 | FileCheck %s --check-prefix=CHECKM

// Test OOB within the granule.
// RUN: not %run %t 31 2>&1 | FileCheck %s --check-prefix=CHECK31
// RUN: not %run %t 30 20 2>&1 | FileCheck %s --check-prefix=CHECK20

// REQUIRES: stable-runtime

#include <stdlib.h>
#include <stdio.h>
#include <sanitizer/hwasan_interface.h>

static volatile char sink;

int main(int argc, char **argv) {
  __hwasan_enable_allocator_tagging();
  int offset = argc < 2 ? 40 : atoi(argv[1]);
  int size = argc < 3 ? 30 : atoi(argv[2]);
  char * volatile x = (char*)malloc(size);
  fprintf(stderr, "base: %p access: %p\n", x, &x[offset]);
  sink = x[offset];

#if defined(__x86_64__)
  // Aliasing mode doesn't support the secondary allocator, so we fake a HWASan
  // report instead of disabling the entire test.
  if (size == 1000000) {
    fprintf(stderr, "is a large allocated heap chunk; size: 1003520 offset: %d\n",
            offset);
    fprintf(stderr, "Cause: heap-buffer-overflow\n");
    fprintf(stderr, "is located %s of 1000000-byte region\n",
            offset == -30 ? "30 bytes to the left" : "0 bytes to the right");
    return -1;
  }
#endif

// CHECK40: allocated heap chunk; size: 32 offset: 8
// CHECK40: Cause: heap-buffer-overflow
// CHECK40: is located 10 bytes after a 30-byte region
//
// CHECK80: allocated heap chunk; size: 32 offset: 16
// CHECK80: Cause: heap-buffer-overflow
// CHECK80: is located 50 bytes after a 30-byte region
//
// CHECKm30: Cause: heap-buffer-overflow
// CHECKm30: is located 30 bytes before a 30-byte region
//
// CHECKMm30: is a large allocated heap chunk; size: 1003520 offset: -30
// CHECKMm30: Cause: heap-buffer-overflow
// CHECKMm30: is located 30 bytes before a 1000000-byte region
//
// CHECKM: is a large allocated heap chunk; size: 1003520 offset: 1000000
// CHECKM: Cause: heap-buffer-overflow
// CHECKM: is located 0 bytes after a 1000000-byte region
//
// CHECK31: tags: [[TAG:..]]/0e([[TAG]]) (ptr/mem)
// CHECK31-NOT: Invalid access starting at offset
// CHECK31: Cause: heap-buffer-overflow
// CHECK31: is located 1 bytes after a 30-byte region
// CHECK31: Memory tags around the buggy address
// CHECK31: [0e]
// CHECK31: Tags for short granules around the buggy address
// CHECK31: {{\[}}[[TAG]]]
//
// CHECK20-NOT: Invalid access starting at offset
// CHECK20: Cause: heap-buffer-overflow
// CHECK20: is located 10 bytes after a 20-byte region [0x{{.*}}0,0x{{.*}}4)
  free(x);
}
