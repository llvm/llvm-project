// Checks that we do not print a faraway buffer overrun if we find a
// use-after-free.
// RUN: %clang_hwasan -O0 %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK
// REQUIRES: stable-runtime

#include <sanitizer/hwasan_interface.h>
#include <stdio.h>
#include <stdlib.h>

#define ALLOC_ATTEMPTS 256

char *Untag(void *x) {
  return (char *)__hwasan_tag_pointer(x, 0);
}

void *FindMatch(void *ptrs[ALLOC_ATTEMPTS], void *value) {
  for (int i = 0; i < ALLOC_ATTEMPTS; ++i) {
    if (!ptrs[i])
      return NULL;
    int distance = Untag(value) - Untag(ptrs[i]);
    // Leave at least one granule of gap to the allocation.
    if (abs(distance) < 1000 && abs(distance) > 32)
      return ptrs[i];
  }
  return NULL;
}

int main(int argc, char **argv) {
  __hwasan_enable_allocator_tagging();
  void *ptrs[ALLOC_ATTEMPTS] = {};
  // Find two allocations that are close enough so that they would be
  // candidates as buffer overflows for each other.
  void *one;
  void *other;
  for (int i = 0; i < ALLOC_ATTEMPTS; ++i) {
    one = malloc(16);
    other = FindMatch(ptrs, one);
    ptrs[i] = one;
    if (other)
      break;
  }
  if (!other) {
    fprintf(stderr, "Could not find closeby allocations.\n");
    abort();
  }
  __hwasan_tag_memory(Untag(one), 3, 16);
  __hwasan_tag_memory(Untag(other), 3, 16);
  // Tag potential adjaceant allocations with a mismatching tag, otherwise this
  // test would flake.
  __hwasan_tag_memory(Untag(one) + 16, 4, 16);
  __hwasan_tag_memory(Untag(one) - 16, 4, 16);
  void *retagged_one = __hwasan_tag_pointer(one, 3);
  free(retagged_one);
  volatile char *ptr = (char *)retagged_one;
  *ptr = 1;
}

// CHECK-NOT: Cause: heap-buffer-overflow
// CHECK: Cause: use-after-free
// CHECK-NOT: Cause: heap-buffer-overflow
