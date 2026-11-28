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

#include "hdr/errno_macros.h"
#include "hdr/types/size_t.h"
#include "hdr/types/wchar_t.h"
#include "hdr/wchar_macros.h"
#include "src/__support/CPP/limits.h"
#include "src/wchar/vswprintf.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

#include <stdarg.h>

using LlvmLibcVSWPrintfTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

int call_vswprintf(wchar_t *__restrict buffer, size_t bufsz,
                   const wchar_t *__restrict format, ...) {
  va_list vlist;
  va_start(vlist, format);
  int ret = LIBC_NAMESPACE::vswprintf(buffer, bufsz, format, vlist);
  va_end(vlist);
  return ret;
}

TEST_F(LlvmLibcVSWPrintfTest, SimpleNoConv) {
  constexpr size_t N = 64;
  wchar_t buff[N];
  int written;

  written = call_vswprintf(buff, N, L"A simple string with no conversions.");
  EXPECT_EQ(written, 36);
  ASSERT_STREQ(buff, L"A simple string with no conversions.");
}

TEST_F(LlvmLibcVSWPrintfTest, PercentConv) {
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

TEST_F(LlvmLibcVSWPrintfTest, CharConv) {
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

  // WEOF is not a valid wide character.
  EXPECT_EQ(call_vswprintf(buff, N, L"%lc", WEOF), -1);
  ASSERT_ERRNO_EQ(EILSEQ);

  // Invalid narrow character (outside ASCII 0..127).
  EXPECT_EQ(call_vswprintf(buff, N, L"%c", 128), -1);
  ASSERT_ERRNO_EQ(EILSEQ);
}

TEST_F(LlvmLibcVSWPrintfTest, CutOff) {
  constexpr size_t N = 100;
  wchar_t buff[N];
  int written;

  written = call_vswprintf(buff, 16, L"A simple string with no conversions.");
  EXPECT_LT(written, 0);
  ASSERT_ERRNO_FAILURE();
  ASSERT_STREQ(buff, L"A simple string");

  written = call_vswprintf(buff, 5, L"%s", "1234567890");
  EXPECT_LT(written, 0);
  ASSERT_ERRNO_FAILURE();
  ASSERT_STREQ(buff, L"1234");

  written = call_vswprintf(buff, 5, L"%ls", L"1234567890");
  EXPECT_LT(written, 0);
  ASSERT_ERRNO_FAILURE();
  ASSERT_STREQ(buff, L"1234");

  written = call_vswprintf(buff, 67, L"%-101c", 'a');
  EXPECT_LT(written, 0);
  ASSERT_ERRNO_FAILURE();
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
  EXPECT_LT(written, 0);
  ASSERT_ERRNO_FAILURE();

  written = call_vswprintf(buff, 5, L"12345");
  EXPECT_LT(written, 0);
  ASSERT_ERRNO_FAILURE();
  ASSERT_STREQ(buff, L"1234");
}

TEST_F(LlvmLibcVSWPrintfTest, NoCutOff) {
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

TEST_F(LlvmLibcVSWPrintfTest, CharsWrittenOverflow) {
  wchar_t buff[1];

  // Trigger an overflow in the return value of vswprintf by writing more than
  // INT_MAX wide characters.
  int int_max = LIBC_NAMESPACE::cpp::numeric_limits<int>::max();
  int written = call_vswprintf(buff, 0, L"%*stest", int_max, "");
  EXPECT_LT(written, 0);
  ASSERT_ERRNO_FAILURE();
}
