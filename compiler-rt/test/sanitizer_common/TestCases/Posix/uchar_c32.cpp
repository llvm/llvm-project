// Test sanitizer_common interception on `mbrtoc32()` and `c32rtomb()`.
//
// RUN: %clangxx -O0 -g %s -o %t && %run %t
// UNSUPPORTED: darwin, android, netbsd, target={{.*(solaris|aix).*}}

#include <assert.h>
#include <locale.h>
#include <stdlib.h>
#include <string.h>
#include <uchar.h>

static bool set_utf8_locale() {
  return (setlocale(LC_CTYPE, "C.UTF-8") ||
          setlocale(LC_CTYPE, "en_US.UTF-8") ||
          setlocale(LC_CTYPE, "en_US.utf8")) &&
         MB_CUR_MAX >= 4;
}

static void test_mbrtoc32_ascii() {
  mbstate_t st = {};
  char32_t c32 = 0xffffffff;
  assert(mbrtoc32(&c32, "z", 1, &st) == 1);
  assert(c32 == U'z');
}

static void test_mbrtoc32_null_char() {
  mbstate_t st = {};
  char32_t c32 = 0xffffffff;
  assert(mbrtoc32(&c32, "", 1, &st) == 0);
  assert(c32 == U'\0');
}

static void test_mbrtoc32_null_pc32() {
  mbstate_t st = {};
  assert(mbrtoc32(nullptr, "y", 1, &st) == 1);
}

static void test_mbrtoc32_null_src() {
  mbstate_t st = {};
  char32_t c32 = U'x';
  assert(mbrtoc32(&c32, nullptr, 10, &st) == 0);
  assert(c32 == U'x');
}

static void test_mbrtoc32_zero_length() {
  mbstate_t st = {};
  char32_t c32 = U'x';
  assert(mbrtoc32(&c32, "z", 0, &st) == (size_t)-2);
  assert(c32 == U'x');
}

static void test_mbrtoc32_utf8_multibyte() {
  if (!set_utf8_locale())
    return;

  const char emoji[4] = {'\xf0', '\x9f', '\xa4', '\xa1'};
  mbstate_t st = {};
  char32_t c32 = 0;
  assert(mbrtoc32(&c32, emoji, sizeof(emoji), &st) == 4);
  assert(c32 == 0x1f921);
}

static void test_mbrtoc32_utf8_incomplete() {
  if (!set_utf8_locale())
    return;

  const char emoji[4] = {'\xf0', '\x9f', '\xa4', '\xa1'};
  mbstate_t st = {};
  char32_t c32 = 0;
  assert(mbrtoc32(&c32, emoji, 1, &st) == (size_t)-2);
  assert(mbrtoc32(&c32, emoji + 1, 1, &st) == (size_t)-2);
  assert(mbrtoc32(&c32, emoji + 2, 1, &st) == (size_t)-2);
  assert(mbrtoc32(&c32, emoji + 3, 1, &st) == 1);
  assert(c32 == 0x1f921);
}

static void test_mbrtoc32_invalid_seq() {
  if (!set_utf8_locale())
    return;

  mbstate_t st = {};
  char32_t c32 = U'x';
  assert(mbrtoc32(&c32, "\xff", 1, &st) == (size_t)-1);
  assert(c32 == U'x');
}

static void test_c32rtomb_ascii() {
  mbstate_t st = {};
  char buf[16] = {};
  assert(c32rtomb(buf, U'z', &st) == 1);
  assert(buf[0] == 'z');
}

static void test_c32rtomb_null_char() {
  mbstate_t st = {};
  char buf[16];
  memset(buf, 'x', sizeof(buf));
  assert(c32rtomb(buf, U'\0', &st) == 1);
  assert(buf[0] == '\0');
}

static void test_c32rtomb_null_dst() {
  mbstate_t st = {};
  assert(c32rtomb(nullptr, U'\0', &st) == 1);
}

static void test_c32rtomb_null_state() {
  char buf[16] = {};
  assert(c32rtomb(nullptr, U'\0', nullptr) == 1);
  assert(c32rtomb(buf, U'z', nullptr) == 1);
  assert(buf[0] == 'z');
}

static void test_c32rtomb_utf8_multibyte() {
  if (!set_utf8_locale())
    return;

  const char emoji[4] = {'\xf0', '\x9f', '\xa4', '\xa1'};
  mbstate_t st = {};
  char buf[16] = {};
  assert(c32rtomb(buf, 0x1f921, &st) == 4);
  assert(memcmp(buf, emoji, 4) == 0);
}

int main() {
  test_mbrtoc32_ascii();
  test_mbrtoc32_null_char();
  test_mbrtoc32_null_pc32();
  test_mbrtoc32_null_src();
  test_mbrtoc32_zero_length();
  test_mbrtoc32_utf8_multibyte();
  test_mbrtoc32_utf8_incomplete();
  test_mbrtoc32_invalid_seq();

  test_c32rtomb_ascii();
  test_c32rtomb_null_char();
  test_c32rtomb_null_dst();
  test_c32rtomb_null_state();
  test_c32rtomb_utf8_multibyte();
  return 0;
}
