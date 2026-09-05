// RUN: %clangxx -O0 %s -o %t && %run %t
// RUN: %clangxx -O2 %s -o %t && %run %t

// REQUIRES: glibc

#include <assert.h>
#include <string.h>

#include "sanitizer_common/sanitizer_specific.h"

extern "C" void *__memcpy_chk(void *dest, const void *src, size_t len,
                              size_t destlen);
extern "C" void *__memmove_chk(void *dest, const void *src, size_t len,
                               size_t destlen);
extern "C" void *__memset_chk(void *dest, int c, size_t len, size_t destlen);
extern "C" void *__mempcpy_chk(void *dest, const void *src, size_t len,
                               size_t destlen);

int main(int argc, char *argv[]) {
  // Test __memcpy_chk basic behavior & shadow propagation
  {
    char src[10];
    char dest[10];
#if __has_feature(memory_sanitizer)
    __msan_unpoison(dest, sizeof(dest));
    __msan_poison(src, sizeof(src));
#endif
    src[1] = 1;
    src[2] = 2;
    __memcpy_chk(dest, src, 5, sizeof(dest));
#if __has_feature(memory_sanitizer)
    // dest[0] is uninitialized
    assert(__msan_test_shadow(dest, 5) == 0);
    // dest[1], dest[2] are initialized, dest[3..4] are uninitialized
    assert(__msan_test_shadow(dest + 1, 4) == 2);
    // dest[5..9] remain initialized
    assert(__msan_test_shadow(dest + 5, 5) == -1);
#else
    assert(dest[1] == 1);
    assert(dest[2] == 2);
#endif
  }

  // Test __memmove_chk basic behavior & shadow propagation
  {
    char src[10];
    char dest[10];
#if __has_feature(memory_sanitizer)
    __msan_unpoison(dest, sizeof(dest));
    __msan_poison(src, sizeof(src));
#endif
    src[1] = 1;
    src[2] = 2;
    __memmove_chk(dest, src, 5, sizeof(dest));
#if __has_feature(memory_sanitizer)
    assert(__msan_test_shadow(dest, 5) == 0);
    assert(__msan_test_shadow(dest + 1, 4) == 2);
    assert(__msan_test_shadow(dest + 5, 5) == -1);
#else
    assert(dest[1] == 1);
    assert(dest[2] == 2);
#endif
  }

  // Test __memset_chk basic behavior & shadow propagation
  {
    char dest[10];
#if __has_feature(memory_sanitizer)
    __msan_poison(dest, sizeof(dest));
#endif
    __memset_chk(dest, 42, 5, sizeof(dest));
#if __has_feature(memory_sanitizer)
    // first 5 bytes are initialized, remaining 5 are uninitialized
    assert(__msan_test_shadow(dest, 10) == 5);
#else
    for (int i = 0; i < 5; ++i) {
      assert(dest[i] == 42);
    }
#endif
  }

  // Test __mempcpy_chk basic behavior & shadow propagation
  {
    char src[10];
    char dest[10];
#if __has_feature(memory_sanitizer)
    __msan_unpoison(dest, sizeof(dest));
    __msan_poison(src, sizeof(src));
#endif
    src[1] = 1;
    src[2] = 2;
    char *res = (char *)__mempcpy_chk(dest, src, 5, sizeof(dest));
    assert(res == dest + 5);
#if __has_feature(memory_sanitizer)
    assert(__msan_test_shadow(dest, 5) == 0);
    assert(__msan_test_shadow(dest + 1, 4) == 2);
    assert(__msan_test_shadow(dest + 5, 5) == -1);
#else
    assert(dest[1] == 1);
    assert(dest[2] == 2);
#endif
  }

  return 0;
}
