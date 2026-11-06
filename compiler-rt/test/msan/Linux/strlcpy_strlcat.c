// REQUIRES: glibc-2.38
// RUN: %clang_msan %s -o %t
// RUN: %run %t
// RUN: not %run %t uninit 2>&1 | FileCheck %s

#include <assert.h>
#include <sanitizer/msan_interface.h>
#include <string.h>

static size_t (*volatile call_strlcpy)(char *, const char *, size_t) = strlcpy;
static size_t (*volatile call_strlcat)(char *, const char *, size_t) = strlcat;

int main(int argc, char **argv) {
  if (argc > 1) {
    char src[8] = "hello";
    char dst[16];

    __msan_poison(src + 2, 1);
    call_strlcpy(dst, src, sizeof(dst));
    return 0;
  }

  char dst[16];
  __msan_poison(dst, sizeof(dst));

  assert(call_strlcpy(dst, "hello", sizeof(dst)) == 5);
  assert(__msan_test_shadow(dst, 6) == -1);
  assert(__msan_test_shadow(dst + 6, 1) == 0);

  assert(call_strlcat(dst, "xy", sizeof(dst)) == 7);
  assert(__msan_test_shadow(dst, 8) == -1);
  assert(__msan_test_shadow(dst + 8, 1) == 0);
  assert(strcmp(dst, "helloxy") == 0);

  __msan_poison(dst, sizeof(dst));
  assert(call_strlcpy(dst, "abcdef", 4) == 6);
  assert(__msan_test_shadow(dst, 4) == -1);
  assert(__msan_test_shadow(dst + 4, 1) == 0);
  assert(strcmp(dst, "abc") == 0);

  char zero_dst[8];
  __msan_poison(zero_dst, sizeof(zero_dst));

  assert(call_strlcpy(zero_dst, "abc", 0) == 3);
  assert(__msan_test_shadow(zero_dst, sizeof(zero_dst)) == 0);

  assert(call_strlcat(zero_dst, "abc", 0) == 3);
  assert(__msan_test_shadow(zero_dst, sizeof(zero_dst)) == 0);

  char unterm_dst[8];
  memcpy(unterm_dst, "abcd", 4);
  __msan_poison(unterm_dst + 4, 4);
  assert(call_strlcat(unterm_dst, "ef", 4) == 6);
  assert(__msan_test_shadow(unterm_dst, 4) == -1);
  assert(__msan_test_shadow(unterm_dst + 4, 4) == 0);

  return 0;
}

// CHECK: Uninitialized bytes in strlcpy at offset 2 inside
