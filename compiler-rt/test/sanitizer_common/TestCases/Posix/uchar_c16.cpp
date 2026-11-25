// Test sanitizer_common interception on `mbrtoc16()` and `c16rtomb()`.
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

static void test_mbrtoc16_ascii() {
  mbstate_t st = {};
  char16_t c16 = 0xffff;
  assert(mbrtoc16(&c16, "a", 1, &st) == 1);
  assert(c16 == u'a');
}

static void test_mbrtoc16_null_char() {
  mbstate_t st = {};
  char16_t c16 = 0xffff;
  assert(mbrtoc16(&c16, "", 1, &st) == 0);
  assert(c16 == u'\0');
}

static void test_mbrtoc16_null_pc16() {
  mbstate_t st = {};
  assert(mbrtoc16(nullptr, "b", 1, &st) == 1);
}

static void test_mbrtoc16_null_src() {
  mbstate_t st = {};
  char16_t c16 = u'x';
  assert(mbrtoc16(&c16, nullptr, 10, &st) == 0);
  assert(c16 == u'x');
}

static void test_mbrtoc16_zero_length() {
  mbstate_t st = {};
  char16_t c16 = u'x';
  assert(mbrtoc16(&c16, "a", 0, &st) == (size_t)-2);
  assert(c16 == u'x');
}

static void test_mbrtoc16_utf8_surrogate_pair() {
  if (!set_utf8_locale())
    return;

  // U+1F921 CLOWN FACE: F0 9F A4 A1 -> UTF-16 surrogates D83E DD21.
  const char emoji[4] = {'\xf0', '\x9f', '\xa4', '\xa1'};
  mbstate_t st = {};
  char16_t c16_hi = 0;
  char16_t c16_lo = 0;
  assert(mbrtoc16(&c16_hi, emoji, sizeof(emoji), &st) == 4);
  assert(c16_hi == 0xd83e);
  assert(mbrtoc16(&c16_lo, emoji + 4, 0, &st) == (size_t)-3);
  assert(c16_lo == 0xdd21);
}

static void test_mbrtoc16_utf8_incomplete() {
  if (!set_utf8_locale())
    return;

  // U+20AC EURO SIGN: E2 82 AC.
  const char euro[3] = {'\xe2', '\x82', '\xac'};
  mbstate_t st = {};
  char16_t c16 = 0;
  assert(mbrtoc16(&c16, euro, 1, &st) == (size_t)-2);
  assert(mbrtoc16(&c16, euro + 1, 1, &st) == (size_t)-2);
  assert(mbrtoc16(&c16, euro + 2, 1, &st) == 1);
  assert(c16 == 0x20ac);
}

static void test_mbrtoc16_invalid_seq() {
  if (!set_utf8_locale())
    return;

  mbstate_t st = {};
  char16_t c16 = u'x';
  assert(mbrtoc16(&c16, "\xff", 1, &st) == (size_t)-1);
  assert(c16 == u'x');
}

static void test_c16rtomb_ascii() {
  mbstate_t st = {};
  char buf[16] = {};
  assert(c16rtomb(buf, u'a', &st) == 1);
  assert(buf[0] == 'a');
}

static void test_c16rtomb_null_char() {
  mbstate_t st = {};
  char buf[16];
  memset(buf, 'x', sizeof(buf));
  assert(c16rtomb(buf, u'\0', &st) == 1);
  assert(buf[0] == '\0');
}

static void test_c16rtomb_null_dst() {
  mbstate_t st = {};
  assert(c16rtomb(nullptr, u'\0', &st) == 1);
}

static void test_c16rtomb_null_state() {
  char buf[16] = {};
  assert(c16rtomb(nullptr, u'\0', nullptr) == 1);
  assert(c16rtomb(buf, u'a', nullptr) == 1);
  assert(buf[0] == 'a');
}

static void test_c16rtomb_utf8_multibyte() {
  if (!set_utf8_locale())
    return;

  // U+20AC EURO SIGN: E2 82 AC.
  mbstate_t st = {};
  char buf[16] = {};
  assert(c16rtomb(buf, 0x20ac, &st) == 3);
  assert(memcmp(buf, "\xe2\x82\xac", 3) == 0);
}

int main() {
  test_mbrtoc16_ascii();
  test_mbrtoc16_null_char();
  test_mbrtoc16_null_pc16();
  test_mbrtoc16_null_src();
  test_mbrtoc16_zero_length();
  test_mbrtoc16_utf8_surrogate_pair();
  test_mbrtoc16_utf8_incomplete();
  test_mbrtoc16_invalid_seq();

  test_c16rtomb_ascii();
  test_c16rtomb_null_char();
  test_c16rtomb_null_dst();
  test_c16rtomb_null_state();
  test_c16rtomb_utf8_multibyte();
  return 0;
}
