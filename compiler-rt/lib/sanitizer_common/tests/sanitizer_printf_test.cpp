//===-- sanitizer_printf_test.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for sanitizer_printf.cpp
//
//===----------------------------------------------------------------------===//
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "gtest/gtest.h"

#include <string.h>
#include <limits.h>

#ifdef __x86_64__
#  include <emmintrin.h>
#endif

namespace __sanitizer {

TEST(Printf, Basic) {
  char buf[1024];
  uptr len = internal_snprintf(
      buf, sizeof(buf), "a%db%zdc%ue%zuf%xh%zxq%pe%sr", (int)-1, (uptr)-2,
      (unsigned)-4, (uptr)5, (unsigned)10, (uptr)11, (void *)0x123, "_string_");
  EXPECT_EQ(len, strlen(buf));

  std::string expectedString = "a-1b-2c4294967292e5fahbq0x";
  expectedString += std::string(SANITIZER_POINTER_FORMAT_LENGTH - 3, '0');
  expectedString += "123e_string_r";
  EXPECT_STREQ(expectedString.c_str(), buf);
}

TEST(Printf, OverflowStr) {
  char buf[] = "123456789";
  uptr len = internal_snprintf(buf, 4, "%s", "abcdef");
  EXPECT_EQ(len, (uptr)6);
  EXPECT_STREQ("abc", buf);
  EXPECT_EQ(buf[3], 0);
  EXPECT_EQ(buf[4], '5');
  EXPECT_EQ(buf[5], '6');
  EXPECT_EQ(buf[6], '7');
  EXPECT_EQ(buf[7], '8');
  EXPECT_EQ(buf[8], '9');
  EXPECT_EQ(buf[9], 0);
}

TEST(Printf, OverflowInt) {
  char buf[] = "123456789";
  internal_snprintf(buf, 4, "%d", -123456789);
  EXPECT_STREQ("-12", buf);
  EXPECT_EQ(buf[3], 0);
  EXPECT_EQ(buf[4], '5');
  EXPECT_EQ(buf[5], '6');
  EXPECT_EQ(buf[6], '7');
  EXPECT_EQ(buf[7], '8');
  EXPECT_EQ(buf[8], '9');
  EXPECT_EQ(buf[9], 0);
}

TEST(Printf, OverflowUint) {
  char buf[] = "123456789";
  uptr val;
  if (sizeof(val) == 4) {
    val = (uptr)0x12345678;
  } else {
    val = (uptr)0x123456789ULL;
  }
  internal_snprintf(buf, 4, "a%zx", val);
  EXPECT_STREQ("a12", buf);
  EXPECT_EQ(buf[3], 0);
  EXPECT_EQ(buf[4], '5');
  EXPECT_EQ(buf[5], '6');
  EXPECT_EQ(buf[6], '7');
  EXPECT_EQ(buf[7], '8');
  EXPECT_EQ(buf[8], '9');
  EXPECT_EQ(buf[9], 0);
}

TEST(Printf, OverflowPtr) {
  char buf[] = "123456789";
  void *p;
  if (sizeof(p) == 4) {
    p = (void*)0x1234567;
  } else {
    p = (void*)0x123456789ULL;
  }
  internal_snprintf(buf, 4, "%p", p);
  EXPECT_STREQ("0x0", buf);
  EXPECT_EQ(buf[3], 0);
  EXPECT_EQ(buf[4], '5');
  EXPECT_EQ(buf[5], '6');
  EXPECT_EQ(buf[6], '7');
  EXPECT_EQ(buf[7], '8');
  EXPECT_EQ(buf[8], '9');
  EXPECT_EQ(buf[9], 0);
}

#if defined(_WIN32)
// Oh well, MSVS headers don't define snprintf.
# define snprintf _snprintf
#endif

template<typename T>
static void TestAgainstLibc(const char *fmt, T arg1, T arg2) {
  char buf[1024];
  uptr len = internal_snprintf(buf, sizeof(buf), fmt, arg1, arg2);
  char buf2[1024];
  snprintf(buf2, sizeof(buf2), fmt, arg1, arg2);
  EXPECT_EQ(len, strlen(buf));
  EXPECT_STREQ(buf2, buf);
}

TEST(Printf, MinMax) {
  TestAgainstLibc<int>("%d-%d", INT_MIN, INT_MAX);
  TestAgainstLibc<unsigned>("%u-%u", 0, UINT_MAX);
  TestAgainstLibc<unsigned>("%x-%x", 0, UINT_MAX);
#if !defined(_WIN32)
  // %z* format doesn't seem to be supported by MSVS.
  TestAgainstLibc<long>("%zd-%zd", LONG_MIN, LONG_MAX);
  TestAgainstLibc<unsigned long>("%zu-%zu", 0, ULONG_MAX);
  TestAgainstLibc<unsigned long>("%zx-%zx", 0, ULONG_MAX);
#endif
}

TEST(Printf, Padding) {
  TestAgainstLibc<int>("%3d - %3d", 1, 0);
  TestAgainstLibc<int>("%3d - %3d", -1, 123);
  TestAgainstLibc<int>("%3d - %3d", -1, -123);
  TestAgainstLibc<int>("%3d - %3d", 12, 1234);
  TestAgainstLibc<int>("%3d - %3d", -12, -1234);
  TestAgainstLibc<int>("%03d - %03d", 1, 0);
  TestAgainstLibc<int>("%03d - %03d", -1, 123);
  TestAgainstLibc<int>("%03d - %03d", -1, -123);
  TestAgainstLibc<int>("%03d - %03d", 12, 1234);
  TestAgainstLibc<int>("%03d - %03d", -12, -1234);
}

TEST(Printf, Precision) {
  char buf[1024];
  uptr len = internal_snprintf(buf, sizeof(buf), "%.*s", 3, "12345");
  EXPECT_EQ(3U, len);
  EXPECT_STREQ("123", buf);
  len = internal_snprintf(buf, sizeof(buf), "%.*s", 6, "12345");
  EXPECT_EQ(5U, len);
  EXPECT_STREQ("12345", buf);
  len = internal_snprintf(buf, sizeof(buf), "%-6s", "12345");
  EXPECT_EQ(6U, len);
  EXPECT_STREQ("12345 ", buf);
  // Check that width does not overflow the smaller buffer, although
  // 10 chars is requested, it stops at the buffer size, 8.
  len = internal_snprintf(buf, 8, "%-10s", "12345");
  EXPECT_EQ(10U, len);  // The required size reported.
  EXPECT_STREQ("12345  ", buf);
}

#ifdef __x86_64__
TEST(Printf, M128) {
  __m128i v = _mm_set_epi32(0x12345678, 0x0a0a0a0a, 0xb0b0b0b0, 0xaabbccdd);
  char buf[128];
  internal_snprintf(buf, sizeof(buf), "%V", PRINTF_128(v));
  EXPECT_STREQ("ddccbbaab0b0b0b00a0a0a0a78563412", buf);
  v = _mm_cvtsi32_si128(0x12345678);
  internal_snprintf(buf, sizeof(buf), "%V", PRINTF_128(v));
  EXPECT_STREQ("78563412000000000000000000000000", buf);
  internal_snprintf(buf, sizeof(buf), "%d %V", 0, PRINTF_128(v));
  EXPECT_STREQ("0 78563412000000000000000000000000", buf);
}
#endif

}  // namespace __sanitizer
