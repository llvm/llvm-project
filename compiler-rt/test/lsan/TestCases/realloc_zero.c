// RUN: %clang_lsan %s -o %t
// RUN: %run %t

#include <assert.h>
#include <stdlib.h>

#if __has_feature(leak_sanitizer)
#  include <sanitizer/allocator_interface.h>
#endif

int main() {
  char *p = malloc(1);
  // The behavior of realloc(p, 0) is implementation-defined.
  // We free the allocation.
  assert(realloc(p, 0) == NULL);

  // realloc(NULL, 0) allocates instead, and must be tracked exactly like
  // malloc(0): a distinct, owned, one-byte allocation. Reporting it as a
  // zero-byte allocation would hide it from the allocator interface.
  void *q = realloc(NULL, 0);
  assert(q != NULL);
#if __has_feature(leak_sanitizer)
  assert(__sanitizer_get_ownership(q));
  assert(__sanitizer_get_allocated_size(q) == 1);
#endif
  free(q);

  p = 0;
}
