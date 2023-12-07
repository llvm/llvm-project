//===-- Unittests for sscanf ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/limits.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/PlatformDefs.h"

#include "src/stdio/sscanf.h"

#include <stdio.h> // For EOF

#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcSScanfTest, SimpleStringConv) {
  int ret_val;
  char buffer[10];
  char buffer2[10];
  ret_val = __llvm_libc::sscanf("abc123", "abc %s", buffer);
  ASSERT_EQ(ret_val, 1);
  ASSERT_STREQ(buffer, "123");

  ret_val = __llvm_libc::sscanf("abc123", "%3s %3s", buffer, buffer2);
  ASSERT_EQ(ret_val, 2);
  ASSERT_STREQ(buffer, "abc");
  ASSERT_STREQ(buffer2, "123");

  ret_val = __llvm_libc::sscanf("abc 123", "%3s%3s", buffer, buffer2);
  ASSERT_EQ(ret_val, 2);
  ASSERT_STREQ(buffer, "abc");
  ASSERT_STREQ(buffer2, "123");
}

TEST(LlvmLibcSScanfTest, IntConvSimple) {
  int ret_val;
  int result = 0;
  ret_val = __llvm_libc::sscanf("123", "%d", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, 123);

  ret_val = __llvm_libc::sscanf("456", "%i", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, 456);

  ret_val = __llvm_libc::sscanf("789", "%x", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, 0x789);

  ret_val = __llvm_libc::sscanf("012", "%o", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, 012);

  ret_val = __llvm_libc::sscanf("345", "%u", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, 345);

  // 288 characters
  ret_val = __llvm_libc::sscanf("10000000000000000000000000000000"
                                "00000000000000000000000000000000"
                                "00000000000000000000000000000000"
                                "00000000000000000000000000000000"
                                "00000000000000000000000000000000"
                                "00000000000000000000000000000000"
                                "00000000000000000000000000000000"
                                "00000000000000000000000000000000"
                                "00000000000000000000000000000000",
                                "%d", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, int(__llvm_libc::cpp::numeric_limits<intmax_t>::max()));

  ret_val = __llvm_libc::sscanf("Not an integer", "%d", &result);
  EXPECT_EQ(ret_val, 0);
}

TEST(LlvmLibcSScanfTest, IntConvLengthModifier) {
  int ret_val;
  uintmax_t max_result = 0;
  int int_result = 0;
  char char_result = 0;

  ret_val = __llvm_libc::sscanf("123", "%ju", &max_result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(max_result, uintmax_t(123));

  // Check overflow handling
  ret_val = __llvm_libc::sscanf("999999999999999999999999999999999999", "%ju",
                                &max_result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(max_result, __llvm_libc::cpp::numeric_limits<uintmax_t>::max());

  // Because this is unsigned, any out of range value should return the maximum,
  // even with a negative sign.
  ret_val = __llvm_libc::sscanf("-999999999999999999999999999999999999", "%ju",
                                &max_result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(max_result, __llvm_libc::cpp::numeric_limits<uintmax_t>::max());

  ret_val = __llvm_libc::sscanf("-18446744073709551616", "%ju", &max_result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(max_result, __llvm_libc::cpp::numeric_limits<uintmax_t>::max());

  // But any number below the maximum should have the - sign applied.
  ret_val = __llvm_libc::sscanf("-1", "%ju", &max_result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(max_result, uintmax_t(-1));

  ret_val = __llvm_libc::sscanf("-1", "%u", &int_result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(int_result, -1);

  max_result = 0xff00ff00ff00ff00;
  char_result = 0x6f;

  // Overflows for sizes larger than the maximum are handled by casting.
  ret_val = __llvm_libc::sscanf("8589967360", "%d", &int_result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(int_result, int(8589967360)); // 2^33 + 2^15

  // Check that the adjacent values weren't touched by the overflow.
  ASSERT_EQ(max_result, uintmax_t(0xff00ff00ff00ff00));
  ASSERT_EQ(char_result, char(0x6f));

  ret_val = __llvm_libc::sscanf("-8589967360", "%d", &int_result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(int_result, int(-8589967360));
  ASSERT_EQ(max_result, uintmax_t(0xff00ff00ff00ff00));
  ASSERT_EQ(char_result, char(0x6f));

  ret_val = __llvm_libc::sscanf("25", "%hhd", &char_result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(char_result, char(25));
}

TEST(LlvmLibcSScanfTest, IntConvBaseSelection) {
  int ret_val;
  int result = 0;
  ret_val = __llvm_libc::sscanf("0xabc123", "%i", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, 0xabc123);

  ret_val = __llvm_libc::sscanf("0456", "%i", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, 0456);

  ret_val = __llvm_libc::sscanf("0999", "%i", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, 0);

  ret_val = __llvm_libc::sscanf("123abc456", "%i", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, 123);
}

TEST(LlvmLibcSScanfTest, IntConvMaxLengthTests) {
  int ret_val;
  int result = 0;

  ret_val = __llvm_libc::sscanf("12", "%1d", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, 1);

  ret_val = __llvm_libc::sscanf("-1", "%1d", &result);
  EXPECT_EQ(ret_val, 0);
  EXPECT_EQ(result, 0);

  ret_val = __llvm_libc::sscanf("+1", "%1d", &result);
  EXPECT_EQ(ret_val, 0);
  EXPECT_EQ(result, 0);

  ret_val = __llvm_libc::sscanf("01", "%1d", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, 0);

  ret_val = __llvm_libc::sscanf("01", "%1i", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, 0);

  ret_val = __llvm_libc::sscanf("0x1", "%2i", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, 0);

  ret_val = __llvm_libc::sscanf("-0x1", "%3i", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, 0);

  ret_val = __llvm_libc::sscanf("-0x123", "%4i", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, -1);

  ret_val = __llvm_libc::sscanf("123456789", "%5i", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, 12345);

  ret_val = __llvm_libc::sscanf("123456789", "%10i", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, 123456789);
}

TEST(LlvmLibcSScanfTest, IntConvNoWriteTests) {
  int ret_val;
  // Result shouldn't be used by these tests, but it's safer to have it and
  // check it.
  int result = 0;
  ret_val = __llvm_libc::sscanf("-1", "%*1d", &result);
  EXPECT_EQ(ret_val, 0);
  EXPECT_EQ(result, 0);

  ret_val = __llvm_libc::sscanf("01", "%*1i", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, 0);

  ret_val = __llvm_libc::sscanf("0x1", "%*2i", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, 0);

  ret_val = __llvm_libc::sscanf("a", "%*i", &result);
  EXPECT_EQ(ret_val, 0);
  EXPECT_EQ(result, 0);

  ret_val = __llvm_libc::sscanf("123", "%*i", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, 0);
}

TEST(LlvmLibcSScanfTest, FloatConvSimple) {
  int ret_val;
  float result = 0;

  float inf = __llvm_libc::fputil::FPBits<float>::inf().get_val();
  float nan = __llvm_libc::fputil::FPBits<float>::build_nan(1);

  ret_val = __llvm_libc::sscanf("123", "%f", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_FP_EQ(result, 123.0);

  ret_val = __llvm_libc::sscanf("456.1", "%a", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_FP_EQ(result, 456.1);

  ret_val = __llvm_libc::sscanf("0x789.ap0", "%e", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_FP_EQ(result, 0x789.ap0);

  ret_val = __llvm_libc::sscanf("0x.8", "%e", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_FP_EQ(result, 0x0.8p0);

  ret_val = __llvm_libc::sscanf("0x8.", "%e", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_FP_EQ(result, 0x8.0p0);

  ret_val = __llvm_libc::sscanf("+12.0e1", "%g", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_FP_EQ(result, 12.0e1);

  ret_val = __llvm_libc::sscanf("inf", "%F", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_FP_EQ(result, inf);

  ret_val = __llvm_libc::sscanf("NaN", "%A", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_FP_EQ(result, nan);

  ret_val = __llvm_libc::sscanf("-InFiNiTy", "%E", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_FP_EQ(result, -inf);

  ret_val = __llvm_libc::sscanf("1e10", "%G", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_FP_EQ(result, 1e10);

  ret_val = __llvm_libc::sscanf(".1", "%G", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_FP_EQ(result, 0.1);

  ret_val = __llvm_libc::sscanf("1.", "%G", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_FP_EQ(result, 1.0);

  ret_val = __llvm_libc::sscanf("0", "%f", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_FP_EQ(result, 0.0);

  ret_val = __llvm_libc::sscanf("Not a float", "%f", &result);
  EXPECT_EQ(ret_val, 0);
}

TEST(LlvmLibcSScanfTest, FloatConvLengthModifier) {
  int ret_val;
  double d_result = 0;
  long double ld_result = 0;

  double d_inf = __llvm_libc::fputil::FPBits<double>::inf().get_val();
  long double ld_nan = __llvm_libc::fputil::FPBits<long double>::build_nan(1);

  ret_val = __llvm_libc::sscanf("123", "%lf", &d_result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_FP_EQ(d_result, 123.0);

  ret_val = __llvm_libc::sscanf("456.1", "%La", &ld_result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_FP_EQ(ld_result, 456.1L);

  ret_val = __llvm_libc::sscanf("inf", "%le", &d_result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_FP_EQ(d_result, d_inf);

  ret_val = __llvm_libc::sscanf("nan", "%Lg", &ld_result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_FP_EQ(ld_result, ld_nan);

  ret_val = __llvm_libc::sscanf("1e-300", "%lF", &d_result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_FP_EQ(d_result, 1e-300);

  ret_val = __llvm_libc::sscanf("1.0e600", "%LA", &ld_result);
  EXPECT_EQ(ret_val, 1);
// 1e600 may be larger than the maximum long double (if long double is double).
// In that case both of these should be evaluated as inf.
#ifdef LONG_DOUBLE_IS_DOUBLE
  EXPECT_FP_EQ(ld_result, d_inf);
#else
  EXPECT_FP_EQ(ld_result, 1.0e600L);
#endif
}

TEST(LlvmLibcSScanfTest, FloatConvLongNumber) {
  int ret_val;
  float result = 0;
  double d_result = 0;

  // 32 characters
  ret_val =
      __llvm_libc::sscanf("123456789012345678901234567890.0", "%f", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_FP_EQ(result, 123456789012345678901234567890.0f);

  // 64 characters
  ret_val = __llvm_libc::sscanf(
      "123456789012345678901234567890123456789012345678901234567890.000", "%la",
      &d_result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_FP_EQ(
      d_result,
      123456789012345678901234567890123456789012345678901234567890.000);

  // 128 characters
  ret_val = __llvm_libc::sscanf(
      "123456789012345678901234567890123456789012345678901234567890"
      "123456789012345678901234567890123456789012345678901234567890.0000000",
      "%le", &d_result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_FP_EQ(
      d_result,
      123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890.0000000);

  // 256 characters
  ret_val = __llvm_libc::sscanf("10000000000000000000000000000000"
                                "00000000000000000000000000000000"
                                "00000000000000000000000000000000"
                                "00000000000000000000000000000000"
                                "00000000000000000000000000000000"
                                "00000000000000000000000000000000"
                                "00000000000000000000000000000000"
                                "00000000000000000000000000000000",
                                "%lf", &d_result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_FP_EQ(d_result, 1e255);

  // 288 characters
  ret_val = __llvm_libc::sscanf("10000000000000000000000000000000"
                                "00000000000000000000000000000000"
                                "00000000000000000000000000000000"
                                "00000000000000000000000000000000"
                                "00000000000000000000000000000000"
                                "00000000000000000000000000000000"
                                "00000000000000000000000000000000"
                                "00000000000000000000000000000000"
                                "00000000000000000000000000000000",
                                "%lf", &d_result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_FP_EQ(d_result, 1e287);
}

TEST(LlvmLibcSScanfTest, FloatConvComplexParsing) {
  int ret_val;
  float result = 0;

  float inf = __llvm_libc::fputil::FPBits<float>::inf().get_val();
  float nan = __llvm_libc::fputil::FPBits<float>::build_nan(1);

  ret_val = __llvm_libc::sscanf("0x1.0e3", "%f", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_FP_EQ(result, 0x1.0e3p0);

  ret_val = __llvm_libc::sscanf("", "%a", &result);
  EXPECT_EQ(ret_val, 0);

  ret_val = __llvm_libc::sscanf("+", "%a", &result);
  EXPECT_EQ(ret_val, 0);

  ret_val = __llvm_libc::sscanf("-", "%a", &result);
  EXPECT_EQ(ret_val, 0);

  ret_val = __llvm_libc::sscanf("+.", "%a", &result);
  EXPECT_EQ(ret_val, 0);

  ret_val = __llvm_libc::sscanf("-.e+10", "%a", &result);
  EXPECT_EQ(ret_val, 0);

  // This is a specific example from the standard. Its behavior diverges from
  // other implementations that accept "100e" as being the same as "100e0"
  ret_val = __llvm_libc::sscanf("100er", "%a", &result);
  EXPECT_EQ(ret_val, 0);

  ret_val = __llvm_libc::sscanf("nah", "%a", &result);
  EXPECT_EQ(ret_val, 0);

  ret_val = __llvm_libc::sscanf("indirection", "%a", &result);
  EXPECT_EQ(ret_val, 0);

  ret_val = __llvm_libc::sscanf("infnan", "%a", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_FP_EQ(result, inf);

  ret_val = __llvm_libc::sscanf("naninf", "%a", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_FP_EQ(result, nan);

  ret_val = __llvm_libc::sscanf("infinityinfinity", "%a", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_FP_EQ(result, inf);

  // For %f to accept a string as representing it has to be either "inf" or
  // "infinity" when it stops. It only stops when it encounters a character that
  // isn't the next one in the string, so it accepts "infi" as the the longest
  // prefix of a possibly valid floating-point number, but determines that it is
  // not valid and returns a matching failure. This is because it can only unget
  // one character so when it finds that the character after the second 'i' is
  // not the next character in "infinity" it can't rewind to the point where it
  // had just "inf".
  ret_val = __llvm_libc::sscanf("infi", "%a", &result);
  EXPECT_EQ(ret_val, 0);

  ret_val = __llvm_libc::sscanf("infinite", "%a", &result);
  EXPECT_EQ(ret_val, 0);

  ret_val = __llvm_libc::sscanf("-.1e1", "%f", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_FP_EQ(result, -.1e1);

  ret_val = __llvm_libc::sscanf("1.2.e1", "%f", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_FP_EQ(result, 1.2);
}

TEST(LlvmLibcSScanfTest, FloatConvMaxWidth) {
  int ret_val;
  float result = 0;

  float inf = __llvm_libc::fputil::FPBits<float>::inf().get_val();

  ret_val = __llvm_libc::sscanf("123", "%3f", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_FP_EQ(result, 123.0);

  ret_val = __llvm_libc::sscanf("123", "%5f", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_FP_EQ(result, 123.0);

  ret_val = __llvm_libc::sscanf("456", "%1f", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_FP_EQ(result, 4.0);

  ret_val = __llvm_libc::sscanf("-789", "%1f", &result);
  EXPECT_EQ(ret_val, 0);

  ret_val = __llvm_libc::sscanf("-123", "%2f", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_FP_EQ(result, -1.0);

  ret_val = __llvm_libc::sscanf("inf", "%2f", &result);
  EXPECT_EQ(ret_val, 0);

  ret_val = __llvm_libc::sscanf("nan", "%1f", &result);
  EXPECT_EQ(ret_val, 0);

  ret_val = __llvm_libc::sscanf("-inf", "%3f", &result);
  EXPECT_EQ(ret_val, 0);

  ret_val = __llvm_libc::sscanf("-nan", "%3f", &result);
  EXPECT_EQ(ret_val, 0);

  // If the max length were not here this would fail as discussed above, but
  // since the max length limits it to the 3 it succeeds.
  ret_val = __llvm_libc::sscanf("infinite", "%3f", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_FP_EQ(result, inf);

  ret_val = __llvm_libc::sscanf("-infinite", "%4f", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_FP_EQ(result, -inf);

  ret_val = __llvm_libc::sscanf("01", "%1f", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_FP_EQ(result, 0.0);

  ret_val = __llvm_libc::sscanf("0x1", "%2f", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_FP_EQ(result, 0.0);

  ret_val = __llvm_libc::sscanf("100e", "%4f", &result);
  EXPECT_EQ(ret_val, 0);

  ret_val = __llvm_libc::sscanf("100e+10", "%5f", &result);
  EXPECT_EQ(ret_val, 0);

  ret_val = __llvm_libc::sscanf("100e10", "%5f", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_FP_EQ(result, 100e1);
}

TEST(LlvmLibcSScanfTest, FloatConvNoWrite) {
  int ret_val;
  float result = 0;

  ret_val = __llvm_libc::sscanf("123", "%*f", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_FP_EQ(result, 0.0);

  ret_val = __llvm_libc::sscanf("456.1", "%*a", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_FP_EQ(result, 0.0);

  ret_val = __llvm_libc::sscanf("0x789.ap0", "%*e", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_FP_EQ(result, 0.0);

  ret_val = __llvm_libc::sscanf("+12.0e1", "%*g", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_FP_EQ(result, 0.0);

  ret_val = __llvm_libc::sscanf("inf", "%*F", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_FP_EQ(result, 0.0);

  ret_val = __llvm_libc::sscanf("NaN", "%*A", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_FP_EQ(result, 0.0);

  ret_val = __llvm_libc::sscanf("-InFiNiTy", "%*E", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_FP_EQ(result, 0.0);

  ret_val = __llvm_libc::sscanf("1e10", "%*G", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_FP_EQ(result, 0.0);

  ret_val = __llvm_libc::sscanf(".1", "%*G", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_FP_EQ(result, 0.0);

  ret_val = __llvm_libc::sscanf("123", "%*3f", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_FP_EQ(result, 0.0);

  ret_val = __llvm_libc::sscanf("123", "%*5f", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_FP_EQ(result, 0.0);

  ret_val = __llvm_libc::sscanf("456", "%*1f", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_FP_EQ(result, 0.0);

  ret_val = __llvm_libc::sscanf("Not a float", "%*f", &result);
  EXPECT_EQ(ret_val, 0);
}

TEST(LlvmLibcSScanfTest, CurPosCombined) {
  int ret_val;
  int result = -1;
  char c_result = 0;

  ret_val = __llvm_libc::sscanf("some text", "%n", &result);
  // %n doesn't count as a conversion for the return value.
  EXPECT_EQ(ret_val, 0);
  EXPECT_EQ(result, 0);

  ret_val = __llvm_libc::sscanf("1234567890", "12345%n", &result);
  EXPECT_EQ(ret_val, 0);
  EXPECT_EQ(result, 5);

  ret_val = __llvm_libc::sscanf("1234567890", "12345%n", &result);
  EXPECT_EQ(ret_val, 0);
  EXPECT_EQ(result, 5);

  // 288 characters
  ret_val = __llvm_libc::sscanf("10000000000000000000000000000000"
                                "00000000000000000000000000000000"
                                "00000000000000000000000000000000"
                                "00000000000000000000000000000000"
                                "00000000000000000000000000000000"
                                "00000000000000000000000000000000"
                                "00000000000000000000000000000000"
                                "00000000000000000000000000000000"
                                "00000000000000000000000000000000",
                                "%*d%hhn", &c_result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(c_result, char(288)); // Overflow is handled by casting.

  // 320 characters
  ret_val = __llvm_libc::sscanf("10000000000000000000000000000000"
                                "00000000000000000000000000000000"
                                "00000000000000000000000000000000"
                                "00000000000000000000000000000000"
                                "00000000000000000000000000000000"
                                "00000000000000000000000000000000"
                                "00000000000000000000000000000000"
                                "00000000000000000000000000000000"
                                "00000000000000000000000000000000"
                                "00000000000000000000000000000000",
                                "%*d%n", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, 320);
}

TEST(LlvmLibcSScanfTest, PointerConvCombined) {
  int ret_val;
  void *result;

  ret_val = __llvm_libc::sscanf("(nullptr)", "%p", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, static_cast<void *>(nullptr));

  ret_val = __llvm_libc::sscanf("(NuLlPtR)", "%p", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, static_cast<void *>(nullptr));

  ret_val = __llvm_libc::sscanf("(NULLPTR)", "%p", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, static_cast<void *>(nullptr));

  ret_val = __llvm_libc::sscanf("(null)", "%p", &result);
  EXPECT_EQ(ret_val, 0);

  ret_val = __llvm_libc::sscanf("(nullptr2", "%p", &result);
  EXPECT_EQ(ret_val, 0);

  ret_val = __llvm_libc::sscanf("0", "%p", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, reinterpret_cast<void *>(0));

  ret_val = __llvm_libc::sscanf("100", "%p", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, reinterpret_cast<void *>(0x100));

  ret_val = __llvm_libc::sscanf("-1", "%p", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, reinterpret_cast<void *>(-1));

  ret_val = __llvm_libc::sscanf("0xabcDEFG", "%p", &result);
  EXPECT_EQ(ret_val, 1);
  EXPECT_EQ(result, reinterpret_cast<void *>(0xabcdef));
}

TEST(LlvmLibcSScanfTest, CombinedConv) {
  int ret_val;
  int result = 0;
  char buffer[10];
  ret_val = __llvm_libc::sscanf("123abc", "%i%s", &result, buffer);
  EXPECT_EQ(ret_val, 2);
  EXPECT_EQ(result, 123);
  ASSERT_STREQ(buffer, "abc");

  ret_val = __llvm_libc::sscanf("0xZZZ", "%i%s", &result, buffer);
  EXPECT_EQ(ret_val, 2);
  EXPECT_EQ(result, 0);
  ASSERT_STREQ(buffer, "ZZZ");

  ret_val = __llvm_libc::sscanf("0xZZZ", "%X%s", &result, buffer);
  EXPECT_EQ(ret_val, 2);
  EXPECT_EQ(result, 0);
  ASSERT_STREQ(buffer, "ZZZ");
}
