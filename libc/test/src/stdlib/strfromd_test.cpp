//===-- Unittests for strfromd --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/macros/properties/architectures.h"
#include "src/stdlib/strfromd.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#define EXPECT_STREQ_LEN(str_size_needed, actual_str, expected_str)            \
  EXPECT_EQ(str_size_needed, static_cast<int>(sizeof(expected_str) - 1));      \
  EXPECT_STREQ(actual_str, expected_str);

struct LlvmLibcStrfromdTest : LIBC_NAMESPACE::testing::ErrnoCheckingTest {};

TEST_F(LlvmLibcStrfromdTest, DecimalFormat) {
  char buff[500];
  int result;

  result = LIBC_NAMESPACE::strfromd(buff, 99, "%f", 1.0);
  EXPECT_STREQ_LEN(result, buff, "1.000000");

  result = LIBC_NAMESPACE::strfromd(buff, 99, "%F", -1.0);
  EXPECT_STREQ_LEN(result, buff, "-1.000000");

  result = LIBC_NAMESPACE::strfromd(buff, 99, "%f", -1.234567);
  EXPECT_STREQ_LEN(result, buff, "-1.234567");

  result = LIBC_NAMESPACE::strfromd(buff, 99, "%f", 0.0);
  EXPECT_STREQ_LEN(result, buff, "0.000000");

  result = LIBC_NAMESPACE::strfromd(buff, 99, "%f", 1.5);
  EXPECT_STREQ_LEN(result, buff, "1.500000");

// Dyadic float is only accurate to ~50 digits, so skip this 300 digit test.
// TODO: Create way to test just the first ~50 digits of a number.
#ifndef LIBC_COPT_FLOAT_TO_STR_REDUCED_PRECISION
  result = LIBC_NAMESPACE::strfromd(buff, 499, "%f", 1e300);
  EXPECT_STREQ_LEN(result, buff,
                   "100000000000000005250476025520442024870446858110815915491"
                   "585411551180245"
                   "798890819578637137508044786404370444383288387817694252323"
                   "536043057564479"
                   "218478670698284838720092657580373783023379478809005936895"
                   "323497079994508"
                   "111903896764088007465274278014249457925878882005684283811"
                   "566947219638686"
                   "5459400540160.000000");
#endif // DLIBC_COPT_FLOAT_TO_STR_REDUCED_PRECISION

  result = LIBC_NAMESPACE::strfromd(buff, 99, "%f", 0.1);
  EXPECT_STREQ_LEN(result, buff, "0.100000");

  result = LIBC_NAMESPACE::strfromd(buff, 99, "%f", 1234567890123456789.0);
  EXPECT_STREQ_LEN(result, buff, "1234567890123456768.000000");

  result = LIBC_NAMESPACE::strfromd(buff, 99, "%f", 9999999999999.99);
  EXPECT_STREQ_LEN(result, buff, "9999999999999.990234");

  result = LIBC_NAMESPACE::strfromd(buff, 99, "%f", 0.1);
  EXPECT_STREQ_LEN(result, buff, "0.100000");

  result = LIBC_NAMESPACE::strfromd(buff, 99, "%f", 1234567890123456789.0);
  EXPECT_STREQ_LEN(result, buff, "1234567890123456768.000000");

  result = LIBC_NAMESPACE::strfromd(buff, 99, "%f", 9999999999999.99);
  EXPECT_STREQ_LEN(result, buff, "9999999999999.990234");

  // Precision Tests
  result = LIBC_NAMESPACE::strfromd(buff, 100, "%.2f", 9999999999999.99);
  EXPECT_STREQ_LEN(result, buff, "9999999999999.99");

  result = LIBC_NAMESPACE::strfromd(buff, 100, "%.1f", 9999999999999.99);
  EXPECT_STREQ_LEN(result, buff, "10000000000000.0");

  result = LIBC_NAMESPACE::strfromd(buff, 100, "%.5f", 1.25);
  EXPECT_STREQ_LEN(result, buff, "1.25000");

  result = LIBC_NAMESPACE::strfromd(buff, 100, "%.0f", 1.25);
  EXPECT_STREQ_LEN(result, buff, "1");

  result = LIBC_NAMESPACE::strfromd(buff, 100, "%.20f", 1.234e-10);
  EXPECT_STREQ_LEN(result, buff, "0.00000000012340000000");
}

TEST_F(LlvmLibcStrfromdTest, HexExponentFormat) {
  char buff[60];
  int result;

  result = LIBC_NAMESPACE::strfromd(buff, 10, "%a", 1.0);
  EXPECT_STREQ_LEN(result, buff, "0x1p+0");

  result = LIBC_NAMESPACE::strfromd(buff, 10, "%A", -1.0);
  EXPECT_STREQ_LEN(result, buff, "-0X1P+0");

  result = LIBC_NAMESPACE::strfromd(buff, 30, "%a", -0x1.abcdef12345p0);
  EXPECT_STREQ_LEN(result, buff, "-0x1.abcdef12345p+0");

  result = LIBC_NAMESPACE::strfromd(buff, 50, "%A", 0x1.abcdef12345p0);
  EXPECT_STREQ_LEN(result, buff, "0X1.ABCDEF12345P+0");

  result = LIBC_NAMESPACE::strfromd(buff, 10, "%a", 0.0);
  EXPECT_STREQ_LEN(result, buff, "0x0p+0");

  result = LIBC_NAMESPACE::strfromd(buff, 40, "%a", 1.0e100);
  EXPECT_STREQ_LEN(result, buff, "0x1.249ad2594c37dp+332");

  result = LIBC_NAMESPACE::strfromd(buff, 30, "%a", 0.1);
  EXPECT_STREQ_LEN(result, buff, "0x1.999999999999ap-4");
}

TEST_F(LlvmLibcStrfromdTest, DecimalExponentFormat) {
  char buff[101];
  int result;

  result = LIBC_NAMESPACE::strfromd(buff, 100, "%e", 1.0);
  EXPECT_STREQ_LEN(result, buff, "1.000000e+00");

  result = LIBC_NAMESPACE::strfromd(buff, 100, "%E", -1.0);
  EXPECT_STREQ_LEN(result, buff, "-1.000000E+00");

  result = LIBC_NAMESPACE::strfromd(buff, 100, "%e", -1.234567);
  EXPECT_STREQ_LEN(result, buff, "-1.234567e+00");

  result = LIBC_NAMESPACE::strfromd(buff, 100, "%e", 0.0);
  EXPECT_STREQ_LEN(result, buff, "0.000000e+00");

  result = LIBC_NAMESPACE::strfromd(buff, 100, "%e", 1.5);
  EXPECT_STREQ_LEN(result, buff, "1.500000e+00");

  result = LIBC_NAMESPACE::strfromd(buff, 100, "%e", 1e300);
  EXPECT_STREQ_LEN(result, buff, "1.000000e+300");

  result = LIBC_NAMESPACE::strfromd(buff, 100, "%e", 1234567890123456789.0);
  EXPECT_STREQ_LEN(result, buff, "1.234568e+18");

  // Precision Tests
  result = LIBC_NAMESPACE::strfromd(buff, 100, "%.1e", 1.0);
  EXPECT_STREQ_LEN(result, buff, "1.0e+00");

  result = LIBC_NAMESPACE::strfromd(buff, 100, "%.1e", 1.99);
  EXPECT_STREQ_LEN(result, buff, "2.0e+00");

  result = LIBC_NAMESPACE::strfromd(buff, 100, "%.1e", 9.99);
  EXPECT_STREQ_LEN(result, buff, "1.0e+01");
}

TEST_F(LlvmLibcStrfromdTest, DecimalAutoFormat) {
  char buff[120];
  int result;

  result = LIBC_NAMESPACE::strfromd(buff, 100, "%g", 1234567890123456789.0);
  EXPECT_STREQ_LEN(result, buff, "1.23457e+18");

  result = LIBC_NAMESPACE::strfromd(buff, 100, "%g", 9999990000000.00);
  EXPECT_STREQ_LEN(result, buff, "9.99999e+12");

  result = LIBC_NAMESPACE::strfromd(buff, 100, "%g", 9999999000000.00);
  EXPECT_STREQ_LEN(result, buff, "1e+13");

  result = LIBC_NAMESPACE::strfromd(buff, 100, "%g", 0xa.aaaaaaaaaaaaaabp-7);
  EXPECT_STREQ_LEN(result, buff, "0.0833333");

  result = LIBC_NAMESPACE::strfromd(buff, 100, "%g", 0.00001);
  EXPECT_STREQ_LEN(result, buff, "1e-05");

  // Precision Tests
  result = LIBC_NAMESPACE::strfromd(buff, 100, "%.0g", 0.0);
  EXPECT_STREQ_LEN(result, buff, "0");

  result = LIBC_NAMESPACE::strfromd(buff, 100, "%.2g", 0.1);
  EXPECT_STREQ_LEN(result, buff, "0.1");

  result = LIBC_NAMESPACE::strfromd(buff, 100, "%.2g", 1.09);
  EXPECT_STREQ_LEN(result, buff, "1.1");

  result = LIBC_NAMESPACE::strfromd(buff, 100, "%.15g", 22.25);
  EXPECT_STREQ_LEN(result, buff, "22.25");

  result = LIBC_NAMESPACE::strfromd(buff, 100, "%.20g", 1.234e-10);
  EXPECT_STREQ_LEN(result, buff, "1.2340000000000000814e-10");
}

TEST_F(LlvmLibcStrfromdTest, InsufficientBufferSize) {
  char buff[20];
  int result;

  result = LIBC_NAMESPACE::strfromd(buff, 5, "%f", 1234567890.0);
  EXPECT_EQ(result, 17);
  ASSERT_STREQ(buff, "1234");

  result = LIBC_NAMESPACE::strfromd(buff, 5, "%.5f", 1.05);
  EXPECT_EQ(result, 7);
  ASSERT_STREQ(buff, "1.05");

  result = LIBC_NAMESPACE::strfromd(buff, 0, "%g", 1.0);
  EXPECT_EQ(result, 1);
  ASSERT_STREQ(buff, "1.05"); // Make sure that buff has not changed
}

TEST_F(LlvmLibcStrfromdTest, InfNanValues) {
  char buff[15];
  int result;

  double inf = LIBC_NAMESPACE::fputil::FPBits<double>::inf().get_val();
  double nan = LIBC_NAMESPACE::fputil::FPBits<double>::quiet_nan().get_val();

  result = LIBC_NAMESPACE::strfromd(buff, 10, "%f", inf);
  EXPECT_STREQ_LEN(result, buff, "inf");

  result = LIBC_NAMESPACE::strfromd(buff, 10, "%A", -inf);
  EXPECT_STREQ_LEN(result, buff, "-INF");

  result = LIBC_NAMESPACE::strfromd(buff, 10, "%f", nan);
  EXPECT_STREQ_LEN(result, buff, "nan");

  result = LIBC_NAMESPACE::strfromd(buff, 10, "%A", -nan);
  EXPECT_STREQ_LEN(result, buff, "-NAN");
}

// https://github.com/llvm/llvm-project/issues/166795
TEST_F(LlvmLibcStrfromdTest, ResultOverflow) {
#ifndef LIBC_TARGET_ARCH_IS_RISCV32
  char buff[100];
  // Trigger an overflow in the return value of strfromd by writing more than
  // INT_MAX bytes.
  int result =
      LIBC_NAMESPACE::strfromd(buff, sizeof(buff), "%.2147483647f", 1.0);

  EXPECT_LT(result, 0);
  ASSERT_ERRNO_FAILURE();
#endif
}
