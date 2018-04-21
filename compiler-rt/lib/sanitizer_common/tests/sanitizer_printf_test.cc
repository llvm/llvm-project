//===-- sanitizer_printf_test.cc ------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Tests for sanitizer_printf.cc
//
//===----------------------------------------------------------------------===//
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "gtest/gtest.h"

#include <string.h>
#include <limits.h>

namespace __sanitizer {

TEST(Printf, Basic) {
  char buf[1024];
  uptr len = internal_snprintf(buf, sizeof(buf),
      "a%db%zdc%ue%zuf%xh%zxq%pe%sr",
      (int)-1, (uptr)-2, // NOLINT
      (unsigned)-4, (uptr)5, // NOLINT
      (unsigned)10, (uptr)11, // NOLINT
      (void*)0x123, "_string_");
  EXPECT_EQ(len, strlen(buf));

  std::string expectedString = "a-1b-2c4294967292e5fahbq0x";
  expectedString += std::string(SANITIZER_POINTER_FORMAT_LENGTH - 3, '0');
  expectedString += "123e_string_r";
  EXPECT_STREQ(expectedString.c_str(), buf);
}

TEST(Printf, OverflowStr) {
  char buf[] = "123456789";
  uptr len = internal_snprintf(buf, 4, "%s", "abcdef");  // NOLINT
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
  internal_snprintf(buf, 4, "%d", -123456789);  // NOLINT
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
  internal_snprintf(buf, 4, "a%zx", val);  // NOLINT
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
  internal_snprintf(buf, 4, "%p", p);  // NOLINT
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
  TestAgainstLibc<int>("%d-%d", INT_MIN, INT_MAX);  // NOLINT
  TestAgainstLibc<unsigned>("%u-%u", 0, UINT_MAX);  // NOLINT
  TestAgainstLibc<unsigned>("%x-%x", 0, UINT_MAX);  // NOLINT
#if !defined(_WIN32)
  // %z* format doesn't seem to be supported by MSVS.
  TestAgainstLibc<long>("%zd-%zd", LONG_MIN, LONG_MAX);  // NOLINT
  TestAgainstLibc<unsigned long>("%zu-%zu", 0, ULONG_MAX);  // NOLINT
  TestAgainstLibc<unsigned long>("%zx-%zx", 0, ULONG_MAX);  // NOLINT
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
  EXPECT_EQ(6U, len);
  EXPECT_STREQ("12345 ", buf);
  // CHeck that precision does not overflow the smaller buffer, although
  // 10 chars is requested, it stops at the buffer size, 8.
  len = internal_snprintf(buf, 8, "%.*s", 10, "12345");
  EXPECT_EQ(10U, len);  // The required size reported.
  EXPECT_STREQ("12345  ", buf);
}

}  // namespace __sanitizer
