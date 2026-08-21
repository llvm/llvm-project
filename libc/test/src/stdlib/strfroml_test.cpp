//===-- Unittests for strfroml --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/macros/properties/architectures.h"
#include "src/stdlib/strfroml.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#define EXPECT_STREQ_LEN(str_size_needed, actual_str, expected_str)            \
  EXPECT_EQ(str_size_needed, static_cast<int>(sizeof(expected_str) - 1));      \
  EXPECT_STREQ(actual_str, expected_str);

struct LlvmLibcStrfromlTest : LIBC_NAMESPACE::testing::ErrnoCheckingTest {};

TEST_F(LlvmLibcStrfromlTest, DecimalFormat) {
  char buff[45];
  int result;

  result = LIBC_NAMESPACE::strfroml(buff, 40, "%f", 1.0L);
  EXPECT_STREQ_LEN(result, buff, "1.000000");

  result = LIBC_NAMESPACE::strfroml(buff, 10, "%.f", -2.5L);
  EXPECT_STREQ_LEN(result, buff, "-2");
}

TEST_F(LlvmLibcStrfromlTest, HexExponentFormat) {
  char buff[55];
  int result;

  result = LIBC_NAMESPACE::strfroml(buff, 50, "%a", 0.1L);
#if defined(LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80)
  EXPECT_STREQ_LEN(result, buff, "0xc.ccccccccccccccdp-7");
#elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT64)
  EXPECT_STREQ_LEN(result, buff, "0x1.999999999999ap-4");
#elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT128)
  EXPECT_STREQ_LEN(result, buff, "0x1.999999999999999999999999999ap-4");
#endif

  result = LIBC_NAMESPACE::strfroml(buff, 20, "%.1a", 0.1L);
#if defined(LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80)
  EXPECT_STREQ_LEN(result, buff, "0xc.dp-7");
#elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT64)
  EXPECT_STREQ_LEN(result, buff, "0x1.ap-4");
#elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT128)
  EXPECT_STREQ_LEN(result, buff, "0x1.ap-4");
#endif

  result = LIBC_NAMESPACE::strfroml(buff, 50, "%a", 1.0e1000L);
#if defined(LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80)
  EXPECT_STREQ_LEN(result, buff, "0xf.38db1f9dd3dac05p+3318");
#elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT64)
  EXPECT_STREQ_LEN(result, buff, "inf");
#elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT128)
  EXPECT_STREQ_LEN(result, buff, "0x1.e71b63f3ba7b580af1a52d2a7379p+3321");
#endif

  result = LIBC_NAMESPACE::strfroml(buff, 50, "%a", 1.0e-1000L);
#if defined(LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80)
  EXPECT_STREQ_LEN(result, buff, "0x8.68a9188a89e1467p-3325");
#elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT64)
  EXPECT_STREQ_LEN(result, buff, "0x0p+0");
#elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT128)
  EXPECT_STREQ_LEN(result, buff, "0x1.0d152311513c28ce202627c06ec2p-3322");
#endif

  result =
      LIBC_NAMESPACE::strfroml(buff, 50, "%.1a", 0xf.fffffffffffffffp16380L);
#if defined(LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80)
  EXPECT_STREQ_LEN(result, buff, "0x1.0p+16384");
#elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT64)
  EXPECT_STREQ_LEN(result, buff, "inf");
#elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT128)
  EXPECT_STREQ_LEN(result, buff, "0x2.0p+16383");
#endif
}

TEST_F(LlvmLibcStrfromlTest, DecimalExponentFormat) {
  // Mark as maybe_unused to silence unused variable
  // warning when long double is not 80-bit
  [[maybe_unused]] char buff[100];
  [[maybe_unused]] int result;

#if defined(LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80)
  result = LIBC_NAMESPACE::strfroml(buff, 90, "%.9e", 1000000000500000000.1L);
  EXPECT_STREQ_LEN(result, buff, "1.000000001e+18");

  result = LIBC_NAMESPACE::strfroml(buff, 90, "%.9e", 1000000000500000000.0L);
  EXPECT_STREQ_LEN(result, buff, "1.000000000e+18");

  result =
      LIBC_NAMESPACE::strfroml(buff, 90, "%e", 0xf.fffffffffffffffp+16380L);
  EXPECT_STREQ_LEN(result, buff, "1.189731e+4932");
#endif // LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80
}

TEST_F(LlvmLibcStrfromlTest, DecimalAutoFormat) {
  // Mark as maybe_unused to silence unused variable
  // warning when long double is not 80-bit
  [[maybe_unused]] char buff[100];
  [[maybe_unused]] int result;

#if defined(LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80)
  result =
      LIBC_NAMESPACE::strfroml(buff, 99, "%g", 0xf.fffffffffffffffp+16380L);
  EXPECT_STREQ_LEN(result, buff, "1.18973e+4932");

  result = LIBC_NAMESPACE::strfroml(buff, 99, "%g", 0xa.aaaaaaaaaaaaaabp-7L);
  EXPECT_STREQ_LEN(result, buff, "0.0833333");

  result = LIBC_NAMESPACE::strfroml(buff, 99, "%g", 9.99999999999e-100L);
  EXPECT_STREQ_LEN(result, buff, "1e-99");
#endif // LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80
}

TEST_F(LlvmLibcStrfromlTest, InsufficientBufferSize) {
  char buff[20];
  int result;

  result = LIBC_NAMESPACE::strfroml(buff, 5, "%f", 1234567890.0l);
  EXPECT_EQ(result, 17);
  ASSERT_STREQ(buff, "1234");

  result = LIBC_NAMESPACE::strfroml(buff, 5, "%.5f", 1.05l);
  EXPECT_EQ(result, 7);
  ASSERT_STREQ(buff, "1.05");

  result = LIBC_NAMESPACE::strfroml(buff, 0, "%g", 1.0l);
  EXPECT_EQ(result, 1);
  ASSERT_STREQ(buff, "1.05"); // Make sure that buff has not changed
}

TEST_F(LlvmLibcStrfromlTest, InfNanValues) {
  char buff[15];
  int result;

  long double inf =
      LIBC_NAMESPACE::fputil::FPBits<long double>::inf().get_val();
  long double nan =
      LIBC_NAMESPACE::fputil::FPBits<long double>::quiet_nan().get_val();

  result = LIBC_NAMESPACE::strfroml(buff, 10, "%f", inf);
  EXPECT_STREQ_LEN(result, buff, "inf");

  result = LIBC_NAMESPACE::strfroml(buff, 10, "%A", -inf);
  EXPECT_STREQ_LEN(result, buff, "-INF");

  result = LIBC_NAMESPACE::strfroml(buff, 10, "%f", nan);
  EXPECT_STREQ_LEN(result, buff, "nan");

  result = LIBC_NAMESPACE::strfroml(buff, 10, "%A", -nan);
  EXPECT_STREQ_LEN(result, buff, "-NAN");
}

// https://github.com/llvm/llvm-project/issues/166795
TEST_F(LlvmLibcStrfromlTest, ResultOverflow) {
#ifndef LIBC_TARGET_ARCH_IS_RISCV32
  char buff[100];
  // Trigger an overflow in the return value of strfroml by writing more than
  // INT_MAX bytes.
  int result =
      LIBC_NAMESPACE::strfroml(buff, sizeof(buff), "%.2147483647f", 1.0);

  EXPECT_LT(result, 0);
  ASSERT_ERRNO_FAILURE();
#endif
}
