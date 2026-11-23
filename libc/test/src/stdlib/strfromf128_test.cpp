//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for strfromf128.
///
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/macros/properties/architectures.h"
#include "src/stdlib/strfromf128.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#define EXPECT_STREQ_LEN(str_size_needed, actual_str, expected_str)            \
  EXPECT_EQ(str_size_needed, static_cast<int>(sizeof(expected_str) - 1));      \
  EXPECT_STREQ(actual_str, expected_str);

namespace {

using LlvmLibcStrfromlTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;
using LIBC_NAMESPACE::fputil::FPBits;

TEST_F(LlvmLibcStrfromlTest, DecimalFormat) {
  char buff[64];
  int result;

  result = LIBC_NAMESPACE::strfromf128(buff, 63, "%f", 1.0);
  EXPECT_STREQ_LEN(result, buff, "1.000000");

  result = LIBC_NAMESPACE::strfromf128(buff, 63, "%.F", -2.5);
  EXPECT_STREQ_LEN(result, buff, "-2");
}

TEST_F(LlvmLibcStrfromlTest, HexExponentFormat) {
  char buff[64];
  int result;

  result = LIBC_NAMESPACE::strfromf128(buff, 63, "%a", 1.0);
  EXPECT_STREQ_LEN(result, buff, "0x1p+0");

  result = LIBC_NAMESPACE::strfromf128(buff, 63, "%A", -1.0);
  EXPECT_STREQ_LEN(result, buff, "-0X1P+0");

  result = LIBC_NAMESPACE::strfromf128(buff, 63, "%a", -0x1.abcdef12345p0);
  EXPECT_STREQ_LEN(result, buff, "-0x1.abcdef12345p+0");

  result = LIBC_NAMESPACE::strfromf128(buff, 63, "%.5A", 0x1.abcdef12345p0);
  EXPECT_STREQ_LEN(result, buff, "0X1.ABCDFP+0");
}

TEST_F(LlvmLibcStrfromlTest, DecimalExponentFormat) {
  char buff[64] = {};
  int result;

  result =
      LIBC_NAMESPACE::strfromf128(buff, 63, "%.9e", 1000000000500000000.1L);
  EXPECT_STREQ_LEN(result, buff, "1.000000001e+18");

  result =
      LIBC_NAMESPACE::strfromf128(buff, 63, "%.9E", 1000000000500000000.0L);
  EXPECT_STREQ_LEN(result, buff, "1.000000000E+18");

  result = LIBC_NAMESPACE::strfromf128(buff, 63, "%e", -1e100L);
  EXPECT_STREQ_LEN(result, buff, "-1.000000e+100");

  result = LIBC_NAMESPACE::strfromf128(buff, 63, "%E", 1.0L);
  EXPECT_STREQ_LEN(result, buff, "1.000000E+00");
}

TEST_F(LlvmLibcStrfromlTest, DecimalAutoFormat) {
  char buff[64] = {};
  int result;

  result = LIBC_NAMESPACE::strfromf128(buff, 63, "%g", 9.99999999999e-100L);
  EXPECT_STREQ_LEN(result, buff, "1e-99");

  result = LIBC_NAMESPACE::strfromf128(buff, 63, "%G", 1e100L);
  EXPECT_STREQ_LEN(result, buff, "1E+100");

  result = LIBC_NAMESPACE::strfromf128(buff, 63, "%g", 1.0);
  EXPECT_STREQ_LEN(result, buff, "1");

  result = LIBC_NAMESPACE::strfromf128(buff, 63, "%g", -0.1);
  EXPECT_STREQ_LEN(result, buff, "-0.1");
}

TEST_F(LlvmLibcStrfromlTest, InsufficientBufferSize) {
  char buff[6] = {'a', 'b', 'c', 'd', 'e', '\0'};
  int result;

  result = LIBC_NAMESPACE::strfromf128(buff, 0, "%g", 1.0);
  EXPECT_EQ(result, 1);
  ASSERT_STREQ(buff, "abcde");

  result = LIBC_NAMESPACE::strfromf128(buff, 5, "%f", 1234567890.0L);
  EXPECT_EQ(result, 17);
  ASSERT_STREQ(buff, "1234");

  result = LIBC_NAMESPACE::strfromf128(buff, 5, "%.5f", 1.05);
  EXPECT_EQ(result, 7);
  ASSERT_STREQ(buff, "1.05");
}

TEST_F(LlvmLibcStrfromlTest, InfNanValues) {
  char buff[64] = {};
  int result;

  float128 inf = FPBits<float128>::inf().get_val();
  float128 nan = FPBits<float128>::quiet_nan().get_val();

  const char *lower_formats[] = {"%f", "%e", "%a", "%g"};
  const char *upper_formats[] = {"%F", "%E", "%A", "%G"};

  for (int i = 0; i < 4; ++i) {
    result = LIBC_NAMESPACE::strfromf128(buff, 63, lower_formats[i], inf);
    EXPECT_STREQ_LEN(result, buff, "inf");
    result = LIBC_NAMESPACE::strfromf128(buff, 63, lower_formats[i], -inf);
    EXPECT_STREQ_LEN(result, buff, "-inf");
    result = LIBC_NAMESPACE::strfromf128(buff, 63, lower_formats[i], nan);
    EXPECT_STREQ_LEN(result, buff, "nan");
    result = LIBC_NAMESPACE::strfromf128(buff, 63, lower_formats[i], -nan);
    EXPECT_STREQ_LEN(result, buff, "-nan");

    result = LIBC_NAMESPACE::strfromf128(buff, 63, upper_formats[i], inf);
    EXPECT_STREQ_LEN(result, buff, "INF");
    result = LIBC_NAMESPACE::strfromf128(buff, 63, upper_formats[i], -inf);
    EXPECT_STREQ_LEN(result, buff, "-INF");
    result = LIBC_NAMESPACE::strfromf128(buff, 63, upper_formats[i], nan);
    EXPECT_STREQ_LEN(result, buff, "NAN");
    result = LIBC_NAMESPACE::strfromf128(buff, 63, upper_formats[i], -nan);
    EXPECT_STREQ_LEN(result, buff, "-NAN");
  }
}

} // namespace
