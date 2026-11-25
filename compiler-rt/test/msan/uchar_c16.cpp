// Test MSan interception on `mbrtoc16()` and `c16rtomb()`.
//
// UNSUPPORTED: darwin, android, netbsd
// RUN: %clangxx_msan -O0 -g %s -o %t
// RUN: %run %t
// RUN: not %run %t src 2>&1 | FileCheck %s
// RUN: not %run %t state 2>&1 | FileCheck %s

#include <assert.h>
#include <locale.h>
#include <sanitizer/msan_interface.h>
#include <stdlib.h>
#include <string.h>
#include <uchar.h>

static bool set_utf8_locale() {
  return (setlocale(LC_CTYPE, "C.UTF-8") ||
          setlocale(LC_CTYPE, "en_US.UTF-8") ||
          setlocale(LC_CTYPE, "en_US.utf8")) &&
         MB_CUR_MAX >= 4;
}

static void test_uninit_src() {
  mbstate_t st = {};
  char src[4];
  __msan_poison(src, sizeof(src));
  char16_t c16;
  (void)mbrtoc16(&c16, src, sizeof(src), &st);
}

static void test_uninit_state() {
  mbstate_t st;
  __msan_poison(&st, sizeof(st));
  char16_t c16;
  (void)mbrtoc16(&c16, "a", 1, &st);
}

// Trailing uninitialized bytes past a complete character must not fault.
static void test_mbrtoc16_partial_init() {
  mbstate_t st = {};
  char partial_init[8];
  __msan_poison(partial_init, sizeof(partial_init));
  partial_init[0] = 'x';

  char16_t c16;
  __msan_poison(&c16, sizeof(c16));
  assert(mbrtoc16(&c16, partial_init, sizeof(partial_init), &st) == 1);
  assert(__msan_test_shadow(&c16, sizeof(c16)) == -1);
  assert(c16 == u'x');

  partial_init[0] = '\0';
  __msan_poison(&c16, sizeof(c16));
  assert(mbrtoc16(&c16, partial_init, sizeof(partial_init), &st) == 0);
  assert(__msan_test_shadow(&c16, sizeof(c16)) == -1);
  assert(c16 == u'\0');
}

// s == nullptr must not write or unpoison *pc16.
static void test_mbrtoc16_null_src() {
  mbstate_t st = {};
  char16_t c16;
  __msan_poison(&c16, sizeof(c16));
  assert(mbrtoc16(&c16, nullptr, 4, &st) == 0);
  assert(__msan_test_shadow(&c16, sizeof(c16)) == 0);
}

// Incomplete sequence (-2) with n == 0 must leave *pc16 poisoned.
static void test_mbrtoc16_zero_length() {
  mbstate_t st = {};
  char16_t c16;
  __msan_poison(&c16, sizeof(c16));
  assert(mbrtoc16(&c16, "a", 0, &st) == (size_t)-2);
  assert(__msan_test_shadow(&c16, sizeof(c16)) == 0);
}

// Encoding error (-1) must leave *pc16 poisoned.
static void test_mbrtoc16_invalid_seq() {
  if (!set_utf8_locale())
    return;

  mbstate_t st = {};
  char16_t c16;
  __msan_poison(&c16, sizeof(c16));
  assert(mbrtoc16(&c16, "\xff", 1, &st) >= (size_t)-2);
  assert(__msan_test_shadow(&c16, sizeof(c16)) == 0);
}

// c16rtomb unpoisons only the written bytes.
static void test_c16rtomb_ascii() {
  mbstate_t st = {};
  char buf[16];
  __msan_poison(buf, sizeof(buf));
  assert(c16rtomb(buf, u'q', &st) == 1);
  assert(__msan_test_shadow(buf, 1) == -1);
  assert(__msan_test_shadow(buf + 1, 1) == 0);
  assert(buf[0] == 'q');
}

static void test_mbrtoc16_utf8_surrogate_pair() {
  if (!set_utf8_locale())
    return;

  const char emoji[4] = {'\xf0', '\x9f', '\xa4', '\xa1'};
  mbstate_t st = {};
  char16_t hi;
  char16_t lo;
  __msan_poison(&hi, sizeof(hi));
  __msan_poison(&lo, sizeof(lo));
  assert(mbrtoc16(&hi, emoji, sizeof(emoji), &st) == 4);
  assert(__msan_test_shadow(&hi, sizeof(hi)) == -1);
  assert(hi == 0xd83e);
  assert(mbrtoc16(&lo, emoji + 4, 0, &st) == (size_t)-3);
  assert(__msan_test_shadow(&lo, sizeof(lo)) == -1);
  assert(lo == 0xdd21);
}

static void test_mbrtoc16_utf8_incomplete() {
  if (!set_utf8_locale())
    return;

  const char euro[3] = {'\xe2', '\x82', '\xac'};
  mbstate_t st = {};
  char16_t c16;
  __msan_poison(&c16, sizeof(c16));
  assert(mbrtoc16(&c16, euro, 2, &st) == (size_t)-2);
  assert(__msan_test_shadow(&c16, sizeof(c16)) == 0);
  assert(mbrtoc16(&c16, euro + 2, 1, &st) == 1);
  assert(__msan_test_shadow(&c16, sizeof(c16)) == -1);
  assert(c16 == 0x20ac);
}

static void test_c16rtomb_utf8_multibyte() {
  if (!set_utf8_locale())
    return;

  mbstate_t st = {};
  char buf[16];
  __msan_poison(buf, sizeof(buf));
  assert(c16rtomb(buf, 0x20ac, &st) == 3);
  assert(__msan_test_shadow(buf, 3) == -1);
  assert(__msan_test_shadow(buf + 3, 1) == 0);
  assert(memcmp(buf, "\xe2\x82\xac", 3) == 0);
}

static void test_mbrtoc16_big5hkscs_continuation() {
  if (!setlocale(LC_CTYPE, "zh_HK.BIG5-HKSCS") &&
      !setlocale(LC_CTYPE, "zh_HK.big5hkscs"))
    return;

  mbstate_t st = {};
  char buf[4];
  __msan_poison(buf, sizeof(buf));
  buf[0] = '\x88';
  buf[1] = '\x62';

  char16_t c1;
  char16_t c2;
  __msan_poison(&c1, sizeof(c1));
  __msan_poison(&c2, sizeof(c2));
  assert(mbrtoc16(&c1, buf, 2, &st) == 2);
  assert(__msan_test_shadow(&c1, sizeof(c1)) == -1);
  assert(c1 == 0xca);

  size_t res = mbrtoc16(&c2, buf + 2, 1, &st);
  assert(res == 0 || res == (size_t)-3);
  assert(__msan_test_shadow(&c2, sizeof(c2)) == -1);
  assert(c2 == 0x304);
}

int main(int argc, char **argv) {
  if (argc > 1) {
    if (strcmp(argv[1], "src") == 0) {
      test_uninit_src();
    } else if (strcmp(argv[1], "state") == 0) {
      test_uninit_state();
    }
    // CHECK: MemorySanitizer: use-of-uninitialized-value
    return 0;
  }

  test_mbrtoc16_partial_init();
  test_mbrtoc16_null_src();
  test_mbrtoc16_zero_length();
  test_mbrtoc16_invalid_seq();
  test_c16rtomb_ascii();
  test_mbrtoc16_utf8_surrogate_pair();
  test_mbrtoc16_utf8_incomplete();
  test_c16rtomb_utf8_multibyte();
  test_mbrtoc16_big5hkscs_continuation();
  return 0;
}
