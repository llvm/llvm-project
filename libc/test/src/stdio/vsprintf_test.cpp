//===-- Unittests for vsprintf --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// These tests are shortened copies of the non-v variants of the printf
// functions. This is because these functions are identical in every way except
// for how the varargs are passed.

#include "src/stdio/vsprintf.h"

#include "test/UnitTest/Test.h"

int call_vsprintf(char *__restrict buffer, const char *__restrict format, ...) {
  va_list vlist;
  va_start(vlist, format);
  int ret = LIBC_NAMESPACE::vsprintf(buffer, format, vlist);
  va_end(vlist);
  return ret;
}

TEST(LlvmLibcVSPrintfTest, SimpleNoConv) {
  char buff[64];
  int written;

  written = call_vsprintf(buff, "A simple string with no conversions.");
  EXPECT_EQ(written, 36);
  ASSERT_STREQ(buff, "A simple string with no conversions.");
}

TEST(LlvmLibcVSPrintfTest, PercentConv) {
  char buff[64];
  int written;

  written = call_vsprintf(buff, "%%");
  EXPECT_EQ(written, 1);
  ASSERT_STREQ(buff, "%");

  written = call_vsprintf(buff, "abc %% def");
  EXPECT_EQ(written, 9);
  ASSERT_STREQ(buff, "abc % def");

  written = call_vsprintf(buff, "%%%%%%");
  EXPECT_EQ(written, 3);
  ASSERT_STREQ(buff, "%%%");
}

TEST(LlvmLibcVSPrintfTest, CharConv) {
  char buff[64];
  int written;

  written = call_vsprintf(buff, "%c", 'a');
  EXPECT_EQ(written, 1);
  ASSERT_STREQ(buff, "a");

  written = call_vsprintf(buff, "%3c %-3c", '1', '2');
  EXPECT_EQ(written, 7);
  ASSERT_STREQ(buff, "  1 2  ");

  written = call_vsprintf(buff, "%*c", 2, '3');
  EXPECT_EQ(written, 2);
  ASSERT_STREQ(buff, " 3");
}
