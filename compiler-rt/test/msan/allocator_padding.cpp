// malloc: all bytes are uninitialized
// RUN: %clang_msan -fsanitize-memory-track-origins=2 %s -o %t && not %run %t 0 2>&1 \
// RUN:     | FileCheck %s --check-prefixes=CHECK,ORIGIN-ALLOC
// RUN: %clang_msan -fsanitize-memory-track-origins=2 %s -o %t && not %run %t 6 2>&1 \
// RUN:     | FileCheck %s --check-prefixes=CHECK,ORIGIN-ALLOC
//
// This test assumes the allocator allocates 16 bytes for malloc(7). Bytes
// 7-15 are padding.
// Edge case: when the origin granularity spans both ALLOC and ALLOC_PADDING,
//            ALLOC takes precedence
// RUN: %clang_msan -fsanitize-memory-track-origins=2 %s -o %t && not %run %t 7 2>&1 \
// RUN:     | FileCheck %s --check-prefixes=CHECK,ORIGIN-ALLOC
//
// Bytes 8-15 are tagged as ALLOC_PADDING.
// RUN: %clang_msan -fsanitize-memory-track-origins=2 %s -o %t && not %run %t 8 2>&1 \
// RUN:     | FileCheck %s --check-prefixes=CHECK,ORIGIN-ALLOC-PADDING
// RUN: %clang_msan -fsanitize-memory-track-origins=2 %s -o %t && not %run %t 15 2>&1 \
// RUN:     | FileCheck %s --check-prefixes=CHECK,ORIGIN-ALLOC-PADDING

// calloc
// Bytes 0-6 are fully initialized, so no MSan report should happen.
// RUN: %clang_msan -fsanitize-memory-track-origins=2 -DUSE_CALLOC %s -o %t && %run %t 0 2>&1
// RUN: %clang_msan -fsanitize-memory-track-origins=2 -DUSE_CALLOC %s -o %t && %run %t 6 2>&1
//
// Byte 7 is uninitialized. Unlike malloc, this is tagged as ALLOC_PADDING
// (since the origin does not need to track bytes 4-6).
// RUN: %clang_msan -fsanitize-memory-track-origins=2 -DUSE_CALLOC %s -o %t && not %run %t 7 2>&1 \
// RUN:     | FileCheck %s --check-prefixes=CHECK,ORIGIN-ALLOC-PADDING
//
// As with malloc, Bytes 8-15 are tagged as ALLOC_PADDING.
// RUN: %clang_msan -fsanitize-memory-track-origins=2 -DUSE_CALLOC %s -o %t && not %run %t 8 2>&1 \
// RUN:     | FileCheck %s --check-prefixes=CHECK,ORIGIN-ALLOC-PADDING
// RUN: %clang_msan -fsanitize-memory-track-origins=2 -DUSE_CALLOC %s -o %t && not %run %t 15 2>&1 \
// RUN:     | FileCheck %s --check-prefixes=CHECK,ORIGIN-ALLOC-PADDING

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
#ifdef USE_CALLOC
  char *p = (char *)calloc(7, 1);
#else
  char *p = (char *)malloc(7);
#endif

  if (argc == 2) {
    int index = atoi(argv[1]);

    printf("p[%d] = %d\n", index, p[index]);
    // CHECK: WARNING: MemorySanitizer: use-of-uninitialized-value
    // CHECK: {{#0 0x.* in main .*allocator_padding.cpp:}}[[@LINE-2]]
    // ORIGIN-ALLOC: Uninitialized value was created by a heap allocation
    // ORIGIN-ALLOC-PADDING: Uninitialized value is outside of heap allocation
    free(p);
  }

  return 0;
}
