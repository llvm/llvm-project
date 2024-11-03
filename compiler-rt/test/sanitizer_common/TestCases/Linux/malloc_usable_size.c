// RUN: %clang -O2 %s -o %t && %run %t

// Must not be implemented, no other reason to install interceptors.
// XFAIL: ubsan

#include <assert.h>
#include <malloc.h>
#include <sanitizer/allocator_interface.h>
#include <stdlib.h>

void *p;

int main() {
  assert(__sanitizer_get_allocated_size(NULL) == 0);
  assert(malloc_usable_size(NULL) == 0);

  int size = 1;
  p = malloc(size);
  assert(__sanitizer_get_allocated_size(p) == size);
  assert(__sanitizer_get_allocated_size_fast(p) == size);
  assert(malloc_usable_size(p) == size);
  free(p);

  size = 1234567;
  p = malloc(size);
  assert(__sanitizer_get_allocated_size(p) == size);
  assert(__sanitizer_get_allocated_size_fast(p) == size);
  assert(malloc_usable_size(p) == size);
  free(p);
  return 0;
}
