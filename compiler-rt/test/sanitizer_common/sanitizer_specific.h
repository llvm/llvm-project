#ifndef __SANITIZER_COMMON_SANITIZER_SPECIFIC_H__
#define __SANITIZER_COMMON_SANITIZER_SPECIFIC_H__

#ifndef __has_feature
#  define __has_feature(x) 0
#endif

#if __has_feature(memory_sanitizer)
#  include <sanitizer/msan_interface.h>
static void check_mem_is_good(void *p, size_t s) {
  __msan_check_mem_is_initialized(p, s);
}
#elif __has_feature(address_sanitizer)
#  include <sanitizer/asan_interface.h>
#  include <stdlib.h>
static void check_mem_is_good(void *p, size_t s) {
  if (__asan_region_is_poisoned(p, s))
    abort();
}
#else
static void check_mem_is_good(void *p, size_t s) {}
#endif

#endif // __SANITIZER_COMMON_SANITIZER_SPECIFIC_H__