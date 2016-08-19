// Test the behavior of malloc/calloc/realloc when the allocation size is huge.
// By default (allocator_may_return_null=0) the process should crash.
// With allocator_may_return_null=1 the allocator should return 0.
//
// RUN: %clangxx_msan -O0 %s -o %t
// RUN: not %run %t malloc 2>&1 | FileCheck %s --check-prefix=CHECK-mCRASH
// RUN: MSAN_OPTIONS=allocator_may_return_null=0 not %run %t malloc 2>&1 | FileCheck %s --check-prefix=CHECK-mCRASH
// RUN: MSAN_OPTIONS=allocator_may_return_null=1     %run %t malloc 2>&1 | FileCheck %s --check-prefix=CHECK-mNULL
// RUN: MSAN_OPTIONS=allocator_may_return_null=0 not %run %t calloc 2>&1 | FileCheck %s --check-prefix=CHECK-cCRASH
// RUN: MSAN_OPTIONS=allocator_may_return_null=1     %run %t calloc 2>&1 | FileCheck %s --check-prefix=CHECK-cNULL
// RUN: MSAN_OPTIONS=allocator_may_return_null=0 not %run %t calloc-overflow 2>&1 | FileCheck %s --check-prefix=CHECK-coCRASH
// RUN: MSAN_OPTIONS=allocator_may_return_null=1     %run %t calloc-overflow 2>&1 | FileCheck %s --check-prefix=CHECK-coNULL
// RUN: MSAN_OPTIONS=allocator_may_return_null=0 not %run %t realloc 2>&1 | FileCheck %s --check-prefix=CHECK-rCRASH
// RUN: MSAN_OPTIONS=allocator_may_return_null=1     %run %t realloc 2>&1 | FileCheck %s --check-prefix=CHECK-rNULL
// RUN: MSAN_OPTIONS=allocator_may_return_null=0 not %run %t realloc-after-malloc 2>&1 | FileCheck %s --check-prefix=CHECK-mrCRASH
// RUN: MSAN_OPTIONS=allocator_may_return_null=1     %run %t realloc-after-malloc 2>&1 | FileCheck %s --check-prefix=CHECK-mrNULL

#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <limits>
int main(int argc, char **argv) {
  volatile size_t size = std::numeric_limits<size_t>::max() - 10000;
  assert(argc == 2);
  char *x = 0;
  if (!strcmp(argv[1], "malloc")) {
    fprintf(stderr, "malloc:\n");
    x = (char*)malloc(size);
  }
  if (!strcmp(argv[1], "calloc")) {
    fprintf(stderr, "calloc:\n");
    x = (char*)calloc(size / 4, 4);
  }

  if (!strcmp(argv[1], "calloc-overflow")) {
    fprintf(stderr, "calloc-overflow:\n");
    volatile size_t kMaxSizeT = std::numeric_limits<size_t>::max();
    size_t kArraySize = 4096;
    volatile size_t kArraySize2 = kMaxSizeT / kArraySize + 10;
    x = (char*)calloc(kArraySize, kArraySize2);
  }

  if (!strcmp(argv[1], "realloc")) {
    fprintf(stderr, "realloc:\n");
    x = (char*)realloc(0, size);
  }
  if (!strcmp(argv[1], "realloc-after-malloc")) {
    fprintf(stderr, "realloc-after-malloc:\n");
    char *t = (char*)malloc(100);
    *t = 42;
    x = (char*)realloc(t, size);
    assert(*t == 42);
  }
  // The NULL pointer is printed differently on different systems, while (long)0
  // is always the same.
  fprintf(stderr, "x: %lx\n", (long)x);
  return x != 0;
}
// CHECK-mCRASH: malloc:
// CHECK-mCRASH: MemorySanitizer's allocator is terminating the process
// CHECK-cCRASH: calloc:
// CHECK-cCRASH: MemorySanitizer's allocator is terminating the process
// CHECK-coCRASH: calloc-overflow:
// CHECK-coCRASH: MemorySanitizer's allocator is terminating the process
// CHECK-rCRASH: realloc:
// CHECK-rCRASH: MemorySanitizer's allocator is terminating the process
// CHECK-mrCRASH: realloc-after-malloc:
// CHECK-mrCRASH: MemorySanitizer's allocator is terminating the process

// CHECK-mNULL: malloc:
// CHECK-mNULL: x: 0
// CHECK-cNULL: calloc:
// CHECK-cNULL: x: 0
// CHECK-coNULL: calloc-overflow:
// CHECK-coNULL: x: 0
// CHECK-rNULL: realloc:
// CHECK-rNULL: x: 0
// CHECK-mrNULL: realloc-after-malloc:
// CHECK-mrNULL: x: 0
