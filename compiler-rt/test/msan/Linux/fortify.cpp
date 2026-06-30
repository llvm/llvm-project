// RUN: %clangxx_msan -O0 %s -o %t && %run %t
// RUN: %clangxx_msan -O2 %s -o %t && %run %t

// REQUIRES: glibc

#include <assert.h>
#include <string.h>
#include <sanitizer/msan_interface.h>

extern "C" void *__memcpy_chk(void *dest, const void *src, size_t len, size_t destlen);
extern "C" void *__memmove_chk(void *dest, const void *src, size_t len, size_t destlen);
extern "C" void *__memset_chk(void *dest, int c, size_t len, size_t destlen);
extern "C" void *__mempcpy_chk(void *dest, const void *src, size_t len, size_t destlen);

int main(int argc, char *argv[]) {
  // Test __memcpy_chk shadow propagation
  {
    char src[10];
    char dest[10];
    __msan_unpoison(dest, sizeof(dest));
    __msan_poison(src, sizeof(src));
    src[1] = 1;
    src[2] = 2;
    __memcpy_chk(dest, src, 5, sizeof(dest));
    assert(__msan_test_shadow(dest, 5) == 0); // dest[0] is uninitialized
    assert(__msan_test_shadow(dest + 1, 4) == 2); // dest[1], dest[2] are initialized, dest[3..4] are uninitialized
    assert(__msan_test_shadow(dest + 5, 5) == -1); // dest[5..9] remain initialized
  }

  // Test __memmove_chk shadow propagation
  {
    char src[10];
    char dest[10];
    __msan_unpoison(dest, sizeof(dest));
    __msan_poison(src, sizeof(src));
    src[1] = 1;
    src[2] = 2;
    __memmove_chk(dest, src, 5, sizeof(dest));
    assert(__msan_test_shadow(dest, 5) == 0);
    assert(__msan_test_shadow(dest + 1, 4) == 2);
    assert(__msan_test_shadow(dest + 5, 5) == -1);
  }

  // Test __memset_chk shadow propagation
  {
    char dest[10];
    __msan_poison(dest, sizeof(dest));
    __memset_chk(dest, 42, 5, sizeof(dest));
    assert(__msan_test_shadow(dest, 10) == 5); // first 5 bytes are initialized, remaining 5 are uninitialized
  }

  // Test __mempcpy_chk shadow propagation
  {
    char src[10];
    char dest[10];
    __msan_unpoison(dest, sizeof(dest));
    __msan_poison(src, sizeof(src));
    src[1] = 1;
    src[2] = 2;
    char *res = (char *)__mempcpy_chk(dest, src, 5, sizeof(dest));
    assert(res == dest + 5);
    assert(__msan_test_shadow(dest, 5) == 0);
    assert(__msan_test_shadow(dest + 1, 4) == 2);
    assert(__msan_test_shadow(dest + 5, 5) == -1);
  }

  return 0;
}
