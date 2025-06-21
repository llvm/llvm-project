// RUN: %clang -std=c23 -O0 %s -o %t && %run %t
// UNSUPPORTED: asan, hwasan, tsan, ubsan

#include <stddef.h>
#include <stdlib.h>

#if defined(__has_feature) && __has_feature(realtime_sanitizer)
#  include <sanitizer/rtsan_interface.h>
#endif

extern void free_sized(void *p, size_t size);

int main() {
#if defined(__has_feature) && __has_feature(realtime_sanitizer)
  __rtsan_disable();
#endif
  volatile void *p = malloc(64);
  free_sized((void *)p, 64);
#if defined(__has_feature) && __has_feature(realtime_sanitizer)
  __rtsan_enable();
#endif
  return 0;
}
