// Test the behavior of malloc/calloc/realloc when the allocation causes OOM
// in the secondary allocator.
// By default (allocator_may_return_null=0) the process should crash.
// With allocator_may_return_null=1 the allocator should return 0.
// Set the limit to 20.5T on 64 bits to account for ASan shadow memory,
// allocator buffers etc. so that the test allocation of ~1T will trigger OOM.
// Limit this test to Linux since we're relying on allocator internal
// limits (shadow memory size, allocation limits etc.)

// RUN: %clangxx_asan -O0 %s -o %t
// RUN: ulimit -v 22024290304
// RUN: not %run %t malloc 2>&1 \
// RUN:   | FileCheck %s --check-prefixes=CHECK-MALLOC,CHECK-CRASH
// RUN: %env_asan_opts=allocator_may_return_null=0 not %run %t malloc 2>&1 \
// RUN:   | FileCheck %s --check-prefixes=CHECK-MALLOC,CHECK-CRASH
// RUN: %env_asan_opts=allocator_may_return_null=1     %run %t malloc 2>&1 \
// RUN:   | FileCheck %s --check-prefixes=CHECK-MALLOC,CHECK-NULL
// RUN: %env_asan_opts=allocator_may_return_null=0 not %run %t calloc 2>&1 \
// RUN:   | FileCheck %s --check-prefixes=CHECK-CALLOC,CHECK-CRASH
// RUN: %env_asan_opts=allocator_may_return_null=1     %run %t calloc 2>&1 \
// RUN:   | FileCheck %s --check-prefixes=CHECK-CALLOC,CHECK-NULL
// RUN: %env_asan_opts=allocator_may_return_null=0 not %run %t realloc 2>&1 \
// RUN:   | FileCheck %s --check-prefixes=CHECK-REALLOC,CHECK-CRASH
// RUN: %env_asan_opts=allocator_may_return_null=1     %run %t realloc 2>&1 \
// RUN:   | FileCheck %s --check-prefixes=CHECK-REALLOC,CHECK-NULL
// RUN: %env_asan_opts=allocator_may_return_null=0 not %run %t realloc-after-malloc 2>&1 \
// RUN:   | FileCheck %s --check-prefixes=CHECK-MALLOC-REALLOC,CHECK-CRASH
// RUN: %env_asan_opts=allocator_may_return_null=1     %run %t realloc-after-malloc 2>&1 \
// RUN:   | FileCheck %s --check-prefixes=CHECK-MALLOC-REALLOC,CHECK-NULL

// ASan shadow memory on s390 is too large for this test.
// AArch64 bots fail on this test.
// TODO(alekseys): Android lit do not run ulimit on device.
// REQUIRES: shadow-scale-3
// UNSUPPORTED: s390,android,aarch64,powerpc64le

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

int main(int argc, char **argv) {
  assert(argc == 2);
  const char *action = argv[1];
  fprintf(stderr, "%s:\n", action);

  // Allocate just a bit less than max allocation size enforced by ASan's
  // allocator (currently 1T and 3G).
  const size_t size =
#if __LP64__
      (1ULL << 40) - (1ULL << 30);
#else
      (3ULL << 30) - (1ULL << 20);
#endif

  void *x = 0;

  if (!strcmp(action, "malloc")) {
    x = malloc(size);
  } else if (!strcmp(action, "calloc")) {
    x = calloc(size / 4, 4);
  } else if (!strcmp(action, "realloc")) {
    x = realloc(0, size);
  } else if (!strcmp(action, "realloc-after-malloc")) {
    char *t = (char*)malloc(100);
    *t = 42;
    x = realloc(t, size);
    assert(*t == 42);
    free(t);
  } else {
    assert(0);
  }

  // The NULL pointer is printed differently on different systems, while (long)0
  // is always the same.
  fprintf(stderr, "x: %lx\n", (long)x);
  free(x);

  return x != 0;
}

// CHECK-MALLOC: malloc:
// CHECK-CALLOC: calloc:
// CHECK-REALLOC: realloc:
// CHECK-MALLOC-REALLOC: realloc-after-malloc:

// CHECK-CRASH: SUMMARY: AddressSanitizer: out-of-memory
// CHECK-NULL: x: 0
