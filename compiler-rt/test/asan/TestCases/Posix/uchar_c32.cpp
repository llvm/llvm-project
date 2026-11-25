// Test ASan interception on `mbrtoc32()` and `c32rtomb()`.
//
// UNSUPPORTED: darwin, android, netbsd, target={{.*(solaris|aix).*}}
// RUN: %clangxx_asan -O0 -g %s -o %t
// RUN: %run %t
// RUN: not %run %t mbrtoc32 2>&1 | FileCheck %s
// RUN: not %run %t c32rtomb 2>&1 | FileCheck %s

#include <assert.h>
#include <locale.h>
#include <stdlib.h>
#include <string.h>
#include <uchar.h>

static void test_mbrtoc32_uaf() {
  mbstate_t st = {};
  char32_t *p = (char32_t *)malloc(sizeof(char32_t));
  free(p);
  mbrtoc32(p, "a", 1, &st);
}

static void test_c32rtomb_uaf() {
  mbstate_t st = {};
  char *p = (char *)malloc(MB_CUR_MAX);
  free(p);
  c32rtomb(p, U'a', &st);
}

// Passing n larger than the 1-byte allocation is valid when 1 byte completes
// the character.
static void test_mbrtoc32_short_alloc() {
  mbstate_t st = {};
  char *one_byte = (char *)malloc(1);
  one_byte[0] = 'a';
  char32_t c32 = 0;
  assert(mbrtoc32(&c32, one_byte, 16, &st) == 1);
  assert(c32 == U'a');
  one_byte[0] = '\0';
  assert(mbrtoc32(&c32, one_byte, 16, &st) == 0);
  assert(c32 == U'\0');
  free(one_byte);
}

// Under zh_HK.BIG5-HKSCS on glibc, "\x88\x62" expands to two code points
// (U+00CA, U+0304) and the second call returns 0 with *pc32 != 0 without
// reading from `s`.
static void test_mbrtoc32_big5hkscs_continuation() {
  if (!setlocale(LC_CTYPE, "zh_HK.BIG5-HKSCS") &&
      !setlocale(LC_CTYPE, "zh_HK.big5hkscs"))
    return;

  mbstate_t st = {};
  char *two_bytes = (char *)malloc(2);
  two_bytes[0] = '\x88';
  two_bytes[1] = '\x62';
  char32_t c1 = 0;
  char32_t c2 = 0;
  assert(mbrtoc32(&c1, two_bytes, 2, &st) == 2);
  assert(c1 == 0xca);
  size_t res = mbrtoc32(&c2, two_bytes + 2, 1, &st);
  assert(res == 0 || res == (size_t)-3);
  assert(c2 == 0x304);
  free(two_bytes);
}

int main(int argc, char **argv) {
  if (argc > 1) {
    if (strcmp(argv[1], "mbrtoc32") == 0) {
      test_mbrtoc32_uaf();
    } else if (strcmp(argv[1], "c32rtomb") == 0) {
      test_c32rtomb_uaf();
    }
    // CHECK: heap-use-after-free
    // CHECK: SUMMARY
    return 0;
  }

  test_mbrtoc32_short_alloc();
  test_mbrtoc32_big5hkscs_continuation();
  return 0;
}
