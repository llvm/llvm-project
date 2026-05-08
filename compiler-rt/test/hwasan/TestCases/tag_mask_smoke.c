// RUN: %clang_hwasan -O0 %s -o %t
// RUN: %env_hwasan_opts=tag_bits=7 %run %t 2>&1

/// Running this once doesn't really prove anything, but it is a smoke test
/// that we don't crash.

#include <sanitizer/hwasan_interface.h>
#include <stdlib.h>

int main() {
  __hwasan_enable_allocator_tagging();
  // DUMP: [alloc] {{.*}} 10{{$}}
  // DUMP: in main{{.*}}malloc_bisect.c
  char *volatile p = (char *)malloc(10);
  if (__hwasan_get_tag_from_pointer(p) & (1 << 7))
    abort();
  free(p);
  __hwasan_disable_allocator_tagging();

  return 0;
}
