// RUN: %clang_hwasan  %s -o %t
// RUN: not %run %t   5  10  26 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK_SMALL,CHECK5
// RUN: not %run %t   7  10  26 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK_SMALL,CHECK7
// RUN: not %run %t   8  20  26 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK_SMALL,CHECK8
// RUN: not %run %t 295 300  26 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK_SMALL,CHECK295
// RUN: not %run %t   1 550 550 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK_SMALL,CHECK1

// Full granule.
// RUN: not %run %t  32  20  26 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK_FULL,CHECK32

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
  int read_offset = atoi(argv[1]);
  int size = atoi(argv[2]);
  int access_size = atoi(argv[3]);
  while (1) {
    char *volatile x = (char *)malloc(size);
    if (__hwasan_test_shadow(x, size + 1) == size)
      memset(x + read_offset, 0, access_size);
    free(x);
  }

  // CHECK_SMALL: WRITE of size {{26|550}} at {{.*}} tags: [[TAG:[0-9a-f]+]]/{{[0-9a-f]+}}([[TAG]]) (ptr/mem)
  // CHECK_FULL: WRITE of size 26 at {{.*}} tags: [[TAG:[0-9a-f]+]]/00 (ptr/mem)

  // CHECK5: Invalid access starting at offset 5
  // CHECK5: is located 5 bytes inside a 10-byte region
  // CHECK7: Invalid access starting at offset 3
  // CHECK7: is located 7 bytes inside a 10-byte region
  // CHECK8: Invalid access starting at offset 12
  // CHECK8: is located 8 bytes inside a 20-byte region
  // CHECK295: Invalid access starting at offset 5
  // CHECK295: is located 295 bytes inside a 300-byte region
  // CHECK1: Invalid access starting at offset 549
  // CHECK1: is located 1 bytes inside a 550-byte region

  // CHECK32-NOT: Invalid access starting at offset
  // CHECK32: is located 12 bytes after a 20-byte region

  // CHECK-LABEL: Memory tags around the buggy address
  // CHECK5: =>{{.*}}[0a]
  // CHECK7: =>{{.*}}[0a]
  // CHECK8: =>{{.*}}[04]
  // CHECK295: =>{{.*}}[0c]
  // CHECK1: =>{{.*}}[06]

  // CHECK32: =>{{.*}}[00]

  // CHECK-LABEL: Tags for short granules around the buggy address
  // CHECK_SMALL: =>{{.*}}{{\[}}[[TAG]]{{\]}}
  // CHECK_FULL: =>{{.*}}[..]
}
