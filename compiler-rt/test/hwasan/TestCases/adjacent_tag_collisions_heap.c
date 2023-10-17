// Test that adjacent heap objects are always tagged differently to prevent unexpected under- and overflows.
// RUN: %clang_hwasan %s -o %t
// RUN: %env_hwasan_opts=random_tags=1,disable_allocator_tagging=0 %run %t

#include <assert.h>
#include <sanitizer/allocator_interface.h>
#include <sanitizer/hwasan_interface.h>
#include <stdio.h>
#include <stdlib.h>

static const size_t sizes[] = {16, 32, 64, 128, 256, 512, 1024, 2048};

void check_collisions_on_heap(size_t size) {
  // Allocate 3 heap objects, which should be placed next to each other
  void *a = malloc(size);
  void *b = malloc(size);
  void *c = malloc(size);

  // Confirm that no object can access adjacent objects
  assert(__hwasan_test_shadow(a, size + 1) != -1);
  assert(__hwasan_test_shadow(b, size + 1) != -1);
  assert(__hwasan_test_shadow(c, size + 1) != -1);

  // Confirm that freeing an object does not increase bounds of objects
  free(b);
  assert(__hwasan_test_shadow(a, size + 1) != -1);
  assert(__hwasan_test_shadow(b, size + 1) != -1);
  assert(__hwasan_test_shadow(c, size + 1) != -1);

  free(a);
  free(c);
}

int main() {
  for (unsigned i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
    for (unsigned j = 0; j < 1000; j++) {
      check_collisions_on_heap(sizes[i]);
    }
  }
  return 0;
}
