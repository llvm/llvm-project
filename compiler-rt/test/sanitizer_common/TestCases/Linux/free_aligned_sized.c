// RUN: %clang -std=c23 -O0 %s -o %t && %run %t
// UNSUPPORTED: asan, hwasan, rtsan, tsan, msan, ubsan

#include <stddef.h>
#include <stdlib.h>

extern void free_aligned_sized(void *p, size_t alignment, size_t size);

int main() {
  volatile void *p = aligned_alloc(128, 1024);
  free_aligned_sized((void *)p, 128, 1024);
  return 0;
}
