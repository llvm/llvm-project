//===-- Unittests for strfromf --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/strfromf.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcStrfromfTest, DecimalFloatFormat) {
  char buff[100];
  int written;

  written = LIBC_NAMESPACE::strfromf(buff, 16, "%f", 1.0);
  EXPECT_EQ(written, 8);
  ASSERT_STREQ(buff, "1.000000");

  written = LIBC_NAMESPACE::strfromf(buff, 20, "%f", 1234567890.0);
  EXPECT_EQ(written, 17);
  ASSERT_STREQ(buff, "1234567936.000000");

  written = LIBC_NAMESPACE::strfromf(buff, 5, "%f", 1234567890.0);
  EXPECT_EQ(written, 17);
  ASSERT_STREQ(buff, "1234");

  written = LIBC_NAMESPACE::strfromf(buff, 67, "%.3f", 1.0);
  EXPECT_EQ(written, 5);
  ASSERT_STREQ(buff, "1.000");

  written = LIBC_NAMESPACE::strfromf(buff, 20, "%1f", 1234567890.0);
  EXPECT_EQ(written, 3);
  ASSERT_STREQ(buff, "%1f");
}

TEST(LlvmLibcStrfromfTest, HexExpFloatFormat) {
  char buff[100];
  int written;

  written = LIBC_NAMESPACE::strfromf(buff, 0, "%a", 1234567890.0);
  EXPECT_EQ(written, 14);

  written = LIBC_NAMESPACE::strfromf(buff, 20, "%a", 1234567890.0);
  EXPECT_EQ(written, 14);
  ASSERT_STREQ(buff, "0x1.26580cp+30");

  written = LIBC_NAMESPACE::strfromf(buff, 20, "%A", 1234567890.0);
  EXPECT_EQ(written, 14);
  ASSERT_STREQ(buff, "0X1.26580CP+30");
}

TEST(LlvmLibcStrfromfTest, DecimalExpFloatFormat) {
  char buff[100];
  int written;
  written = LIBC_NAMESPACE::strfromf(buff, 20, "%.9e", 1234567890.0);
  EXPECT_EQ(written, 15);
  ASSERT_STREQ(buff, "1.234567936e+09");

  written = LIBC_NAMESPACE::strfromf(buff, 20, "%.9E", 1234567890.0);
  EXPECT_EQ(written, 15);
  ASSERT_STREQ(buff, "1.234567936E+09");
}

TEST(LlvmLibcStrfromfTest, AutoDecimalFloatFormat) {
  char buff[100];
  int written;

  written = LIBC_NAMESPACE::strfromf(buff, 20, "%.9g", 1234567890.0);
  EXPECT_EQ(written, 14);
  ASSERT_STREQ(buff, "1.23456794e+09");

  written = LIBC_NAMESPACE::strfromf(buff, 20, "%.9G", 1234567890.0);
  EXPECT_EQ(written, 14);
  ASSERT_STREQ(buff, "1.23456794E+09");

  written = LIBC_NAMESPACE::strfromf(buff, 0, "%G", 1.0);
  EXPECT_EQ(written, 1);
}

TEST(LlvmLibcStrfromfTest, ImproperFormatString) {

  char buff[100];
  int retval;
  retval = LIBC_NAMESPACE::strfromf(
      buff, 37, "A simple string with no conversions.", 1.0);
  EXPECT_EQ(retval, 36);
  ASSERT_STREQ(buff, "A simple string with no conversions.");

  retval = LIBC_NAMESPACE::strfromf(
      buff, 37, "%A simple string with one conversion, should overwrite.", 1.0);
  EXPECT_EQ(retval, 6);
  ASSERT_STREQ(buff, "0X1P+0");

  retval = LIBC_NAMESPACE::strfromf(buff, 74,
                                    "A simple string with one conversion in %A "
                                    "between, writes string as it is",
                                    1.0);
  EXPECT_EQ(retval, 73);
  ASSERT_STREQ(buff, "A simple string with one conversion in %A between, "
                     "writes string as it is");

  retval = LIBC_NAMESPACE::strfromf(buff, 36,
                                    "A simple string with one conversion", 1.0);
  EXPECT_EQ(retval, 35);
  ASSERT_STREQ(buff, "A simple string with one conversion");
}
