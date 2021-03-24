// Test the behavior of malloc/calloc/realloc/new when the allocation size
// exceeds the HWASan allocator's max allowed one.
// By default (allocator_may_return_null=0) the process should crash. With
// allocator_may_return_null=1 the allocator should return 0 and set errno to
// the appropriate error code.
//
// RUN: %clangxx_hwasan -O0 %s -o %t
// RUN: not %run %t malloc 2>&1 | FileCheck %s --check-prefix=CHECK-mCRASH
// RUN: %env_hwasan_opts=allocator_may_return_null=0 not %run %t malloc 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-mCRASH
// RUN: %env_hwasan_opts=allocator_may_return_null=1     %run %t malloc 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-mNULL
// RUN: %env_hwasan_opts=allocator_may_return_null=0 not %run %t calloc 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-cCRASH
// RUN: %env_hwasan_opts=allocator_may_return_null=1     %run %t calloc 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-cNULL
// RUN: %env_hwasan_opts=allocator_may_return_null=0 not %run %t calloc-overflow 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-coCRASH
// RUN: %env_hwasan_opts=allocator_may_return_null=1     %run %t calloc-overflow 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-coNULL
// RUN: %env_hwasan_opts=allocator_may_return_null=0 not %run %t realloc 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-rCRASH
// RUN: %env_hwasan_opts=allocator_may_return_null=1     %run %t realloc 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-rNULL
// RUN: %env_hwasan_opts=allocator_may_return_null=0 not %run %t realloc-after-malloc 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-mrCRASH
// RUN: %env_hwasan_opts=allocator_may_return_null=1     %run %t realloc-after-malloc 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-mrNULL
// RUN: %env_hwasan_opts=allocator_may_return_null=0 not %run %t new 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-nCRASH
// RUN: %env_hwasan_opts=allocator_may_return_null=1 not %run %t new 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-nCRASH-OOM
// RUN: %env_hwasan_opts=allocator_may_return_null=0 not %run %t new-nothrow 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-nnCRASH
// RUN: %env_hwasan_opts=allocator_may_return_null=1     %run %t new-nothrow 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-nnNULL

// REQUIRES: stable-runtime

// TODO(alekseyshl): Fix it.
// UNSUPPORTED: android

#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits>
#include <new>

int main(int argc, char **argv) {
  assert(argc == 2);
  const char *action = argv[1];
  fprintf(stderr, "%s:\n", action);

  static const size_t kMaxAllowedMallocSizePlusOne = (1UL << 40) + 1;

  void *x = nullptr;
  if (!strcmp(action, "malloc")) {
    x = malloc(kMaxAllowedMallocSizePlusOne);
  } else if (!strcmp(action, "calloc")) {
    x = calloc((kMaxAllowedMallocSizePlusOne / 4) + 1, 4);
  } else if (!strcmp(action, "calloc-overflow")) {
    volatile size_t kMaxSizeT = std::numeric_limits<size_t>::max();
    size_t kArraySize = 4096;
    volatile size_t kArraySize2 = kMaxSizeT / kArraySize + 10;
    x = calloc(kArraySize, kArraySize2);
  } else if (!strcmp(action, "realloc")) {
    x = realloc(0, kMaxAllowedMallocSizePlusOne);
  } else if (!strcmp(action, "realloc-after-malloc")) {
    char *t = (char*)malloc(100);
    *t = 42;
    x = realloc(t, kMaxAllowedMallocSizePlusOne);
    assert(*t == 42);
    free(t);
  } else if (!strcmp(action, "new")) {
    x = operator new(kMaxAllowedMallocSizePlusOne);
  } else if (!strcmp(action, "new-nothrow")) {
    x = operator new(kMaxAllowedMallocSizePlusOne, std::nothrow);
  } else {
    assert(0);
  }

  fprintf(stderr, "errno: %d\n", errno);

  free(x);

  return x != nullptr;
}

// CHECK-mCRASH: malloc:
// CHECK-mCRASH: SUMMARY: HWAddressSanitizer: allocation-size-too-big
// CHECK-cCRASH: calloc:
// CHECK-cCRASH: SUMMARY: HWAddressSanitizer: allocation-size-too-big
// CHECK-coCRASH: calloc-overflow:
// CHECK-coCRASH: SUMMARY: HWAddressSanitizer: calloc-overflow
// CHECK-rCRASH: realloc:
// CHECK-rCRASH: SUMMARY: HWAddressSanitizer: allocation-size-too-big
// CHECK-mrCRASH: realloc-after-malloc:
// CHECK-mrCRASH: SUMMARY: HWAddressSanitizer: allocation-size-too-big
// CHECK-nCRASH: new:
// CHECK-nCRASH: SUMMARY: HWAddressSanitizer: allocation-size-too-big
// CHECK-nCRASH-OOM: new:
// CHECK-nCRASH-OOM: SUMMARY: HWAddressSanitizer: out-of-memory
// CHECK-nnCRASH: new-nothrow:
// CHECK-nnCRASH: SUMMARY: HWAddressSanitizer: allocation-size-too-big

// CHECK-mNULL: malloc:
// CHECK-mNULL: errno: 12
// CHECK-cNULL: calloc:
// CHECK-cNULL: errno: 12
// CHECK-coNULL: calloc-overflow:
// CHECK-coNULL: errno: 12
// CHECK-rNULL: realloc:
// CHECK-rNULL: errno: 12
// CHECK-mrNULL: realloc-after-malloc:
// CHECK-mrNULL: errno: 12
// CHECK-nnNULL: new-nothrow:
