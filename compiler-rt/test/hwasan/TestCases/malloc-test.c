// Test basic malloc functionality.
// RUN: %clang_hwasan %s -o %t
// RUN: %run %t

#include <stdlib.h>
#include <assert.h>
#include <sanitizer/hwasan_interface.h>
#include <sanitizer/allocator_interface.h>

int main() {
  __hwasan_enable_allocator_tagging();
  char *a1 = (char*)malloc(0);
  assert(a1 != 0);
  assert(__sanitizer_get_allocated_size(a1) == 0);
  free(a1);
}
