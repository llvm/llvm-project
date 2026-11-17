//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for vswprintf.
///
//===----------------------------------------------------------------------===//

// These tests are shortened copies of the non-v variants of the swprintf
// functions. This is because these functions are identical in every way except
// for how the varargs are passed.

#include "hdr/types/size_t.h"
#include "hdr/types/wchar_t.h"
#include "src/wchar/vswprintf.h"
#include "test/UnitTest/Test.h"

#include <stdarg.h>

int call_vswprintf(wchar_t *__restrict buffer, size_t bufsz,
                   const wchar_t *__restrict format, ...) {
  va_list vlist;
  va_start(vlist, format);
  int ret = LIBC_NAMESPACE::vswprintf(buffer, bufsz, format, vlist);
  va_end(vlist);
  return ret;
}

TEST(LlvmLibcVSWPrintfTest, SimpleNoConv) {
  constexpr size_t N = 64;
  wchar_t buff[N];
  int written;

  written = call_vswprintf(buff, N, L"A simple string with no conversions.");
  EXPECT_EQ(written, 36);
  ASSERT_STREQ(buff, L"A simple string with no conversions.");
}

TEST(LlvmLibcVSWPrintfTest, PercentConv) {
  constexpr size_t N = 64;
  wchar_t buff[N];
  int written;

  written = call_vswprintf(buff, N, L"%%");
  EXPECT_EQ(written, 1);
  ASSERT_STREQ(buff, L"%");

  written = call_vswprintf(buff, N, L"abc %% def");
  EXPECT_EQ(written, 9);
  ASSERT_STREQ(buff, L"abc % def");

  written = call_vswprintf(buff, N, L"%%%%%%");
  EXPECT_EQ(written, 3);
  ASSERT_STREQ(buff, L"%%%");
}

TEST(LlvmLibcVSWPrintfTest, CharConv) {
  constexpr size_t N = 64;
  wchar_t buff[N];
  int written;

  written = call_vswprintf(buff, N, L"%c", 'a');
  EXPECT_EQ(written, 1);
  ASSERT_STREQ(buff, L"a");

  written = call_vswprintf(buff, N, L"%3c %-3c", '1', '2');
  EXPECT_EQ(written, 7);
  ASSERT_STREQ(buff, L"  1 2  ");

  written = call_vswprintf(buff, N, L"%*c", 2, '3');
  EXPECT_EQ(written, 2);
  ASSERT_STREQ(buff, L" 3");
}

TEST(LlvmLibcVSWPrintfTest, CutOff) {
  constexpr size_t N = 100;
  wchar_t buff[N];
  int written;

  written = call_vswprintf(buff, 16, L"A simple string with no conversions.");
  EXPECT_EQ(written, 36);
  ASSERT_STREQ(buff, L"A simple string");

  written = call_vswprintf(buff, 5, L"%s", "1234567890");
  EXPECT_EQ(written, 10);
  ASSERT_STREQ(buff, L"1234");

  written = call_vswprintf(buff, 5, L"%ls", L"1234567890");
  EXPECT_EQ(written, 10);
  ASSERT_STREQ(buff, L"1234");

  written = call_vswprintf(buff, 67, L"%-101c", 'a');
  EXPECT_EQ(written, 101);
  ASSERT_STREQ(buff, L"a "
                     L"        " // Each of these is 8 spaces, and there are 8.
                     L"        " // In total there are 65 spaces
                     L"        " // L'a' + 65 spaces + L'\0' = 67
                     L"        "
                     L"        "
                     L"        "
                     L"        "
                     L"        ");

  // passing null as the output pointer is allowed as long as bufsz is 0.
  written = call_vswprintf(nullptr, 0, L"%s and more", "1234567890");
  EXPECT_EQ(written, 19);
}

TEST(LlvmLibcVSWPrintfTest, NoCutOff) {
  constexpr size_t N = 64;
  wchar_t buff[N];
  int written;

  written = call_vswprintf(buff, 37, L"A simple string with no conversions.");
  EXPECT_EQ(written, 36);
  ASSERT_STREQ(buff, L"A simple string with no conversions.");

  written = call_vswprintf(buff, 20, L"%s", "1234567890");
  EXPECT_EQ(written, 10);
  ASSERT_STREQ(buff, L"1234567890");

  written = call_vswprintf(buff, 20, L"%ls", L"1234567890");
  EXPECT_EQ(written, 10);
  ASSERT_STREQ(buff, L"1234567890");
}
