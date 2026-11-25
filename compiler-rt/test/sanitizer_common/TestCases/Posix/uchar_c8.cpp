// Test sanitizer_common interception on `mbrtoc8()` and `c8rtomb()`.
//
// REQUIRES: have-uchar-c8
// RUN: %clangxx -std=c++20 -O0 -g %s -o %t && %run %t

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

static void test_mbrtoc8_ascii() {
  mbstate_t st = {};
  char8_t c8 = 0xff;
  assert(mbrtoc8(&c8, "a", 1, &st) == 1);
  assert(c8 == u8'a');
}

static void test_mbrtoc8_null_char() {
  mbstate_t st = {};
  char8_t c8 = 0xff;
  assert(mbrtoc8(&c8, "", 1, &st) == 0);
  assert(c8 == u8'\0');
}

static void test_mbrtoc8_null_pc8() {
  mbstate_t st = {};
  assert(mbrtoc8(nullptr, "b", 1, &st) == 1);
}

static void test_mbrtoc8_null_src() {
  mbstate_t st = {};
  char8_t c8 = u8'x';
  assert(mbrtoc8(&c8, nullptr, 10, &st) == 0);
  assert(c8 == u8'x');
}

static void test_mbrtoc8_zero_length() {
  mbstate_t st = {};
  char8_t c8 = u8'x';
  assert(mbrtoc8(&c8, "a", 0, &st) == (size_t)-2);
  assert(c8 == u8'x');
}

static void test_mbrtoc8_utf8_continuation() {
  if (!set_utf8_locale())
    return;

  const char emoji[4] = {'\xf0', '\x9f', '\xa4', '\xa1'};
  mbstate_t st = {};
  char8_t out[4] = {};
  assert(mbrtoc8(&out[0], emoji, sizeof(emoji), &st) == 4);
  assert(mbrtoc8(&out[1], emoji + 4, 0, &st) == (size_t)-3);
  assert(mbrtoc8(&out[2], emoji + 4, 0, &st) == (size_t)-3);
  assert(mbrtoc8(&out[3], emoji + 4, 0, &st) == (size_t)-3);
  assert(memcmp(out, emoji, 4) == 0);
}

static void test_mbrtoc8_utf8_incomplete() {
  if (!set_utf8_locale())
    return;

  const char emoji[4] = {'\xf0', '\x9f', '\xa4', '\xa1'};
  mbstate_t st = {};
  char8_t out[4] = {};
  assert(mbrtoc8(&out[0], emoji, 1, &st) == (size_t)-2);
  assert(mbrtoc8(&out[0], emoji + 1, 1, &st) == (size_t)-2);
  assert(mbrtoc8(&out[0], emoji + 2, 1, &st) == (size_t)-2);
  assert(mbrtoc8(&out[0], emoji + 3, 1, &st) == 1);
  assert(mbrtoc8(&out[1], emoji + 4, 0, &st) == (size_t)-3);
  assert(mbrtoc8(&out[2], emoji + 4, 0, &st) == (size_t)-3);
  assert(mbrtoc8(&out[3], emoji + 4, 0, &st) == (size_t)-3);
  assert(memcmp(out, emoji, 4) == 0);
}

static void test_mbrtoc8_invalid_seq() {
  if (!set_utf8_locale())
    return;

  mbstate_t st = {};
  char8_t c8 = u8'x';
  assert(mbrtoc8(&c8, "\xff", 1, &st) == (size_t)-1);
  assert(c8 == u8'x');
}

static void test_c8rtomb_ascii() {
  mbstate_t st = {};
  char buf[16] = {};
  assert(c8rtomb(buf, u8'a', &st) == 1);
  assert(buf[0] == 'a');
}

static void test_c8rtomb_null_char() {
  mbstate_t st = {};
  char buf[16];
  memset(buf, 'x', sizeof(buf));
  assert(c8rtomb(buf, u8'\0', &st) == 1);
  assert(buf[0] == '\0');
}

static void test_c8rtomb_null_dst() {
  mbstate_t st = {};
  assert(c8rtomb(nullptr, u8'\0', &st) == 1);
}

static void test_c8rtomb_null_state() {
  char buf[16] = {};
  assert(c8rtomb(nullptr, u8'\0', nullptr) == 1);
  assert(c8rtomb(buf, u8'a', nullptr) == 1);
  assert(buf[0] == 'a');
}

static void test_c8rtomb_utf8_multibyte() {
  if (!set_utf8_locale())
    return;

  const char emoji[4] = {'\xf0', '\x9f', '\xa4', '\xa1'};
  mbstate_t st = {};
  char buf[16] = {};
  assert(c8rtomb(buf, static_cast<char8_t>(0xf0), &st) == 0);
  assert(c8rtomb(buf, static_cast<char8_t>(0x9f), &st) == 0);
  assert(c8rtomb(buf, static_cast<char8_t>(0xa4), &st) == 0);
  assert(c8rtomb(buf, static_cast<char8_t>(0xa1), &st) == 4);
  assert(memcmp(buf, emoji, 4) == 0);
}

int main() {
  test_mbrtoc8_ascii();
  test_mbrtoc8_null_char();
  test_mbrtoc8_null_pc8();
  test_mbrtoc8_null_src();
  test_mbrtoc8_zero_length();
  test_mbrtoc8_utf8_continuation();
  test_mbrtoc8_utf8_incomplete();
  test_mbrtoc8_invalid_seq();

  test_c8rtomb_ascii();
  test_c8rtomb_null_char();
  test_c8rtomb_null_dst();
  test_c8rtomb_null_state();
  test_c8rtomb_utf8_multibyte();
  return 0;
}
