// Test that disabling/enabling tagging does not trigger false reports on
// allocations happened in a different state.

// RUN: %clang_hwasan -O1 %s -o %t && %run %t 2>&1

#include <assert.h>
#include <sanitizer/hwasan_interface.h>
#include <stdlib.h>

enum {
  COUNT = 5,
  SZ = 10,
};
void *x[COUNT];

int main() {
  __hwasan_enable_allocator_tagging();
  for (unsigned i = 0; i < COUNT; ++i) {
    x[i] = malloc(SZ);
    assert(__hwasan_test_shadow(x[i], SZ) == -1);
  }
  for (unsigned i = 0; i < COUNT; ++i)
    free(x[i]);

  __hwasan_disable_allocator_tagging();
  for (unsigned i = 0; i < COUNT; ++i) {
    x[i] = malloc(SZ);
    assert(__hwasan_tag_pointer(x[i], 0) == x[i]);
    assert(__hwasan_test_shadow(x[i], SZ) == -1);
  }
  for (unsigned i = 0; i < COUNT; ++i)
    free(x[i]);

  __hwasan_enable_allocator_tagging();
  for (unsigned i = 0; i < COUNT; ++i) {
    x[i] = malloc(SZ);
    assert(__hwasan_test_shadow(x[i], SZ) == -1);
  }
  for (unsigned i = 0; i < COUNT; ++i)
    free(x[i]);
  return 0;
}
