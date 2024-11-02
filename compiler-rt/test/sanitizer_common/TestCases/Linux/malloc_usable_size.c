// RUN: %clang -O2 %s -o %t && %run %t

// Ubsan does not provide allocator.
// UNSUPPORTED: ubsan

#include <assert.h>
#include <malloc.h>
#include <sanitizer/allocator_interface.h>
#include <stdlib.h>

int main() {
  assert(__sanitizer_get_allocated_size(NULL) == 0);
  assert(malloc_usable_size(NULL) == 0);

  int size = 1234567;
  void *p = malloc(size);
  assert(__sanitizer_get_allocated_size(p) == size);
  assert(malloc_usable_size(p) == size);
  free(p);
  return 0;
}
