// Regression test for the Win64 path in asan_allocator's Reallocate where
// realloc()-ing a chunk that was originally created by malloc(0) (and
// internally upgraded to a 1-byte allocation whose single byte is shadow-
// poisoned by asan_mark_zero_allocation) used to spuriously report a
// heap-buffer-overflow when the CRT aliased memcpy and memmove. See
// https://github.com/llvm/llvm-project/pull/193633 and
// https://github.com/llvm/llvm-project/issues/126077.
//
// RUN: %clang_cl_asan %Od %s %Fe%t
// RUN: %run %t 2>&1 | FileCheck %s
//
// Also exercise the static CRT linkage which is the configuration that
// originally surfaced the issue on the buildbots.
// RUN: %clang_cl_asan %Od %MT %s %Fe%t
// RUN: %run %t 2>&1 | FileCheck %s

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
  // malloc(0) returns a non-null pointer to a 1-byte chunk whose single
  // user-visible byte is intentionally poisoned. realloc()-ing it must not
  // attempt to copy that byte through the shadow-checked path.
  void *p = malloc(0);
  if (!p)
    return 1;

  void *q = realloc(p, 32);
  if (!q)
    return 2;

  // Use the new allocation to make sure it is a real, writable region.
  memset(q, 'a', 32);

  free(q);
  fprintf(stderr, "OK\n");
  return 0;
}

// CHECK-NOT: AddressSanitizer
// CHECK-NOT: heap-buffer-overflow
// CHECK: OK
