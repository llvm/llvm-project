//===-- Unittests for strfromf --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/macros/properties/architectures.h"
#include "src/stdlib/strfromf.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#define EXPECT_STREQ_LEN(str_size_needed, actual_str, expected_str)            \
  EXPECT_EQ(str_size_needed, static_cast<int>(sizeof(expected_str) - 1));      \
  EXPECT_STREQ(actual_str, expected_str);

using LlvmLibcStrfromfTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcStrfromfTest, DecimalFormat) {
  char buff[70];
  int result;

  result = LIBC_NAMESPACE::strfromf(buff, 16, "%f", 1.0f);
  EXPECT_STREQ_LEN(result, buff, "1.000000");

  result = LIBC_NAMESPACE::strfromf(buff, 20, "%f", 1234567890.0f);
  EXPECT_STREQ_LEN(result, buff, "1234567936.000000");

  result = LIBC_NAMESPACE::strfromf(buff, 67, "%.3f", 1.0f);
  EXPECT_STREQ_LEN(result, buff, "1.000");
}

TEST_F(LlvmLibcStrfromfTest, HexExponentFormat) {
  char buff[25];
  int result;

  result = LIBC_NAMESPACE::strfromf(buff, 0, "%a", 1234567890.0f);
  EXPECT_EQ(result, 14);

  result = LIBC_NAMESPACE::strfromf(buff, 20, "%a", 1234567890.0f);
  EXPECT_EQ(result, 14);
  ASSERT_STREQ(buff, "0x1.26580cp+30");

  result = LIBC_NAMESPACE::strfromf(buff, 20, "%A", 1234567890.0f);
  EXPECT_EQ(result, 14);
  ASSERT_STREQ(buff, "0X1.26580CP+30");
}

TEST_F(LlvmLibcStrfromfTest, DecimalExponentFormat) {
  char buff[25];
  int result;

  result = LIBC_NAMESPACE::strfromf(buff, 20, "%.9e", 1234567890.0f);
  EXPECT_STREQ_LEN(result, buff, "1.234567936e+09");

  result = LIBC_NAMESPACE::strfromf(buff, 20, "%.9E", 1234567890.0f);
  EXPECT_STREQ_LEN(result, buff, "1.234567936E+09");
}

TEST_F(LlvmLibcStrfromfTest, DecimalAutoFormat) {
  char buff[25];
  int result;

  result = LIBC_NAMESPACE::strfromf(buff, 20, "%.9g", 1234567890.0f);
  EXPECT_STREQ_LEN(result, buff, "1.23456794e+09");

  result = LIBC_NAMESPACE::strfromf(buff, 20, "%.9G", 1234567890.0f);
  EXPECT_STREQ_LEN(result, buff, "1.23456794E+09");
}

TEST_F(LlvmLibcStrfromfTest, InsufficientBufferSize) {
  char buff[20];
  int result;

  result = LIBC_NAMESPACE::strfromf(buff, 5, "%f", 1234567890.0f);
  EXPECT_EQ(result, 17);
  ASSERT_STREQ(buff, "1234");

  result = LIBC_NAMESPACE::strfromf(buff, 5, "%.5f", 1.05f);
  EXPECT_EQ(result, 7);
  ASSERT_STREQ(buff, "1.05");

  result = LIBC_NAMESPACE::strfromf(buff, 0, "%g", 1.0f);
  EXPECT_EQ(result, 1);
  ASSERT_STREQ(buff, "1.05"); // Make sure that buff has not changed
}

// TODO: fix https://github.com/llvm/llvm-project/issues/217708.
#if 0
TEST_F(LlvmLibcStrfromfTest, InfNanValues) {
  char buff[15];
  int result;

  float inf = LIBC_NAMESPACE::fputil::FPBits<float>::inf().get_val();
  float nan = LIBC_NAMESPACE::fputil::FPBits<float>::quiet_nan().get_val();

  result = LIBC_NAMESPACE::strfromf(buff, 10, "%f", inf);
  EXPECT_STREQ_LEN(result, buff, "inf");

  result = LIBC_NAMESPACE::strfromf(buff, 10, "%A", -inf);
  EXPECT_STREQ_LEN(result, buff, "-INF");

  result = LIBC_NAMESPACE::strfromf(buff, 10, "%f", nan);
  EXPECT_STREQ_LEN(result, buff, "nan");

  result = LIBC_NAMESPACE::strfromf(buff, 10, "%A", -nan);
  EXPECT_STREQ_LEN(result, buff, "-NAN");
}
#endif

// https://github.com/llvm/llvm-project/issues/166795
TEST_F(LlvmLibcStrfromfTest, ResultOverflow) {
#ifndef LIBC_TARGET_ARCH_IS_RISCV32
  char buff[100];
  // Trigger an overflow in the return value of strfromf by writing more than
  // INT_MAX bytes.
  int result =
      LIBC_NAMESPACE::strfromf(buff, sizeof(buff), "%.2147483647f", 1.0f);

  EXPECT_LT(result, 0);
  ASSERT_ERRNO_FAILURE();
#endif
}
