// Test basic new functionality.
// RUN: %clangxx_hwasan %s -o %t
// RUN: %run %t

#include <stdlib.h>
#include <assert.h>
#include <sanitizer/hwasan_interface.h>
#include <sanitizer/allocator_interface.h>

int main() {
  __hwasan_enable_allocator_tagging();

  size_t volatile n = 0;
  char *a1 = new char[n];
  assert(a1 != nullptr);
  assert(__sanitizer_get_allocated_size(a1) == 0);
  delete[] a1;
}
