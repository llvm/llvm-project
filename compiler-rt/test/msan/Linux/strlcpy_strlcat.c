// REQUIRES: glibc-2.38
// RUN: %clang_msan -O0 -g -fno-builtin -fPIE -pie %s -o %t
// RUN: env MSAN_OPTIONS=strict_string_checks=0 %run %t
// RUN: env MSAN_OPTIONS=strict_string_checks=1 %run %t
// RUN: not env MSAN_OPTIONS=strict_string_checks=0 %run %t uninit 2>&1 | FileCheck %s

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

  assert(call_strlcpy(dst, "hello", sizeof(dst)) == 5);
  assert(__msan_test_shadow(dst, 6) == -1);

  assert(call_strlcat(dst, "xy", sizeof(dst)) == 7);
  assert(__msan_test_shadow(dst, 8) == -1);
  assert(strcmp(dst, "helloxy") == 0);

  return 0;
}

// CHECK: Uninitialized bytes in strlcpy at offset 2 inside
