// Test the behavior of malloc/calloc/realloc/new when the allocation size
// exceeds the sanitizer's allocator max allowed one.
// By default (allocator_may_return_null=0) the process should crash. With
// allocator_may_return_null=1 the allocator should return nullptr and set errno
// to the appropriate error code.
//
// RUN: %clangxx -O0 %s -o %t
// RUN: not %run %t malloc 2>&1 | FileCheck %s --check-prefix=CHECK-mCRASH
// RUN: %env_tool_opts=allocator_may_return_null=0 not %run %t malloc 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-mCRASH
// RUN: %env_tool_opts=allocator_may_return_null=1     %run %t malloc 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NULL
// RUN: %env_tool_opts=allocator_may_return_null=0 not %run %t calloc 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-cCRASH
// RUN: %env_tool_opts=allocator_may_return_null=1     %run %t calloc 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NULL
// RUN: %env_tool_opts=allocator_may_return_null=0 not %run %t calloc-overflow 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-coCRASH
// RUN: %env_tool_opts=allocator_may_return_null=1     %run %t calloc-overflow 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NULL
// RUN: %env_tool_opts=allocator_may_return_null=0 not %run %t realloc 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-rCRASH
// RUN: %env_tool_opts=allocator_may_return_null=1     %run %t realloc 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NULL
// RUN: %env_tool_opts=allocator_may_return_null=0 not %run %t realloc-after-malloc 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-mrCRASH
// RUN: %env_tool_opts=allocator_may_return_null=1     %run %t realloc-after-malloc 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NULL
// RUN: %env_tool_opts=allocator_may_return_null=0 not %run %t new 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-nCRASH
// RUN: %env_tool_opts=allocator_may_return_null=1 not %run %t new 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-nCRASH-OOM
// RUN: %env_tool_opts=allocator_may_return_null=0 not %run %t new-nothrow 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-nnCRASH
// RUN: %env_tool_opts=allocator_may_return_null=1     %run %t new-nothrow 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NULL

// TODO(alekseyshl): win32 is disabled due to failing errno tests, fix it there.
// UNSUPPORTED: ubsan, target={{.*windows-msvc.*}}

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

  // The maximum value of all supported sanitizers (search for
  // kMaxAllowedMallocSize). For ASan + LSan, ASan limit is used.
  static const size_t kMaxAllowedMallocSizePlusOne =
#if __LP64__ || defined(_WIN64)
      (1ULL << 40) + 1;
#else
      (3UL << 30) + 1;
#endif

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

  // The NULL pointer is printed differently on different systems, while (long)0
  // is always the same.
  fprintf(stderr, "errno: %d, x: %lx\n", errno, (long)x);

  return 0;
}

// CHECK-mCRASH: malloc:
// CHECK-mCRASH: {{SUMMARY: .*Sanitizer: allocation-size-too-big}}
// CHECK-cCRASH: calloc:
// CHECK-cCRASH: {{SUMMARY: .*Sanitizer: allocation-size-too-big}}
// CHECK-coCRASH: calloc-overflow:
// CHECK-coCRASH: {{SUMMARY: .*Sanitizer: calloc-overflow}}
// CHECK-rCRASH: realloc:
// CHECK-rCRASH: {{SUMMARY: .*Sanitizer: allocation-size-too-big}}
// CHECK-mrCRASH: realloc-after-malloc:
// CHECK-mrCRASH: {{SUMMARY: .*Sanitizer: allocation-size-too-big}}
// CHECK-nCRASH: new:
// CHECK-nCRASH: {{SUMMARY: .*Sanitizer: allocation-size-too-big}}
// CHECK-nCRASH-OOM: new:
// CHECK-nCRASH-OOM: {{SUMMARY: .*Sanitizer: out-of-memory}}
// CHECK-nnCRASH: new-nothrow:
// CHECK-nnCRASH: {{SUMMARY: .*Sanitizer: allocation-size-too-big}}

// CHECK-NULL: {{malloc|calloc|calloc-overflow|realloc|realloc-after-malloc|new-nothrow}}
// CHECK-NULL: errno: 12, x: 0
