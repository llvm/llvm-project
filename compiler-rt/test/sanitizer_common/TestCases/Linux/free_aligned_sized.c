// RUN: %clang -std=c23 -O0 %s -o %t && %run %t
// UNSUPPORTED: asan, hwasan, ubsan

#include <stddef.h>
#include <stdlib.h>

extern void *aligned_alloc(size_t alignment, size_t size);

extern void free_aligned_sized(void *p, size_t alignment, size_t size);

int main() {
  volatile void *p = aligned_alloc(128, 1024);
  free_aligned_sized((void *)p, 128, 1024);
  return 0;
}
