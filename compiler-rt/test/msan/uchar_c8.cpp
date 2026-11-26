// Test MSan interception on `mbrtoc8()` and `c8rtomb()`.
//
// REQUIRES: have-uchar-c8
// RUN: %clangxx_msan -std=c++20 -O0 -g %s -o %t
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
  char8_t c8;
  (void)mbrtoc8(&c8, src, sizeof(src), &st);
}

static void test_uninit_state() {
  mbstate_t st;
  __msan_poison(&st, sizeof(st));
  char8_t c8;
  (void)mbrtoc8(&c8, "a", 1, &st);
}

static void test_mbrtoc8_partial_init() {
  mbstate_t st = {};
  char partial_init[8];
  __msan_poison(partial_init, sizeof(partial_init));
  partial_init[0] = 'x';

  char8_t c8;
  __msan_poison(&c8, sizeof(c8));
  assert(mbrtoc8(&c8, partial_init, sizeof(partial_init), &st) == 1);
  assert(__msan_test_shadow(&c8, sizeof(c8)) == -1);
  assert(c8 == u8'x');

  partial_init[0] = '\0';
  __msan_poison(&c8, sizeof(c8));
  assert(mbrtoc8(&c8, partial_init, sizeof(partial_init), &st) == 0);
  assert(__msan_test_shadow(&c8, sizeof(c8)) == -1);
  assert(c8 == u8'\0');
}

static void test_mbrtoc8_null_src() {
  mbstate_t st = {};
  char8_t c8;
  __msan_poison(&c8, sizeof(c8));
  assert(mbrtoc8(&c8, nullptr, 4, &st) == 0);
  assert(__msan_test_shadow(&c8, sizeof(c8)) == 0);
}

static void test_mbrtoc8_zero_length() {
  mbstate_t st = {};
  char8_t c8;
  __msan_poison(&c8, sizeof(c8));
  assert(mbrtoc8(&c8, "a", 0, &st) == (size_t)-2);
  assert(__msan_test_shadow(&c8, sizeof(c8)) == 0);
}

static void test_c8rtomb_ascii() {
  mbstate_t st = {};
  char buf[16];
  __msan_poison(buf, sizeof(buf));
  assert(c8rtomb(buf, u8'q', &st) == 1);
  assert(__msan_test_shadow(buf, 1) == -1);
  assert(__msan_test_shadow(buf + 1, 1) == 0);
  assert(buf[0] == 'q');
}

static void test_mbrtoc8_utf8_continuation() {
  if (!set_utf8_locale())
    return;

  const char emoji[4] = {'\xf0', '\x9f', '\xa4', '\xa1'};
  mbstate_t st = {};
  char8_t out[4];
  __msan_poison(out, sizeof(out));
  assert(mbrtoc8(&out[0], emoji, sizeof(emoji), &st) == 4);
  assert(__msan_test_shadow(&out[0], 1) == -1);
  assert(__msan_test_shadow(&out[1], 1) == 0);
  assert(mbrtoc8(&out[1], emoji + 4, 0, &st) == (size_t)-3);
  assert(mbrtoc8(&out[2], emoji + 4, 0, &st) == (size_t)-3);
  assert(mbrtoc8(&out[3], emoji + 4, 0, &st) == (size_t)-3);
  assert(__msan_test_shadow(out, 4) == -1);
  assert(memcmp(out, emoji, 4) == 0);
}

static void test_c8rtomb_utf8_multibyte() {
  if (!set_utf8_locale())
    return;

  const char emoji[4] = {'\xf0', '\x9f', '\xa4', '\xa1'};
  mbstate_t st = {};
  char buf[16];
  __msan_poison(buf, sizeof(buf));
  assert(c8rtomb(buf, static_cast<char8_t>(0xf0), &st) == 0);
  assert(__msan_test_shadow(buf, 1) == 0);
  assert(c8rtomb(buf, static_cast<char8_t>(0x9f), &st) == 0);
  assert(c8rtomb(buf, static_cast<char8_t>(0xa4), &st) == 0);
  assert(c8rtomb(buf, static_cast<char8_t>(0xa1), &st) == 4);
  assert(__msan_test_shadow(buf, 4) == -1);
  assert(__msan_test_shadow(buf + 4, 1) == 0);
  assert(memcmp(buf, emoji, 4) == 0);
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

  test_mbrtoc8_partial_init();
  test_mbrtoc8_null_src();
  test_mbrtoc8_zero_length();
  test_c8rtomb_ascii();
  test_mbrtoc8_utf8_continuation();
  test_c8rtomb_utf8_multibyte();
  return 0;
}
