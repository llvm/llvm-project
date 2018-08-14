// Test allocator aliases.
//
// RUN: %clangxx_hwasan -O0 %s -o %t && %run %t

#include <sanitizer/hwasan_interface.h>

int main() {
  void *volatile sink;
  sink = (void *)&__sanitizer_posix_memalign;
  sink = (void *)&__sanitizer_memalign;
  sink = (void *)&__sanitizer_aligned_alloc;
  sink = (void *)&__sanitizer___libc_memalign;
  sink = (void *)&__sanitizer_valloc;
  sink = (void *)&__sanitizer_pvalloc;
  sink = (void *)&__sanitizer_free;
  sink = (void *)&__sanitizer_cfree;
  sink = (void *)&__sanitizer_malloc_usable_size;
  sink = (void *)&__sanitizer_mallinfo;
  sink = (void *)&__sanitizer_mallopt;
  sink = (void *)&__sanitizer_malloc_stats;
  sink = (void *)&__sanitizer_calloc;
  sink = (void *)&__sanitizer_realloc;
  sink = (void *)&__sanitizer_malloc;

  // sanity check
  void *p = __sanitizer_malloc(100);
  p = __sanitizer_realloc(p, 200);
  __sanitizer_free(p);
}
