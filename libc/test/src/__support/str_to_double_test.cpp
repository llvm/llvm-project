//===-- Unittests for str_to_float<double> --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/macros/config.h"
#include "str_to_fp_test.h"

namespace LIBC_NAMESPACE_DECL {
using LlvmLibcStrToDblTest = LlvmLibcStrToFloatTest<double>;

TEST_F(LlvmLibcStrToDblTest, ClingerFastPathFloat64Simple) {
  clinger_fast_path_test(123, 0, 0x1EC00000000000, 1029);
  clinger_fast_path_test(1234567890123456, 1, 0x15ee2a2eb5a5c0, 1076);
  clinger_fast_path_test(1234567890, -10, 0x1f9add3739635f, 1019);
}

TEST_F(LlvmLibcStrToDblTest, ClingerFastPathFloat64ExtendedExp) {
  clinger_fast_path_test(1, 30, 0x193e5939a08cea, 1122);
  clinger_fast_path_test(1, 37, 0x1e17b84357691b, 1145);
  clinger_fast_path_fails_test(10, 37);
  clinger_fast_path_fails_test(1, 100);
}

TEST_F(LlvmLibcStrToDblTest, ClingerFastPathFloat64NegativeExp) {
  clinger_fast_path_test(1, -10, 0x1b7cdfd9d7bdbb, 989);
  clinger_fast_path_test(1, -20, 0x179ca10c924223, 956);
  clinger_fast_path_fails_test(1, -25);
}

TEST_F(LlvmLibcStrToDblTest, EiselLemireFloat64Simple) {
  eisel_lemire_test(12345678901234567890u, 1, 0x1AC53A7E04BCDA, 1089);
  eisel_lemire_test(123, 0, 0x1EC00000000000, 1029);
  eisel_lemire_test(12345678901234568192u, 0, 0x156A95319D63E2, 1086);
}

TEST_F(LlvmLibcStrToDblTest, EiselLemireFloat64SpecificFailures) {
  // These test cases have caused failures in the past.
  eisel_lemire_test(358416272, -33, 0x1BBB2A68C9D0B9, 941);
  eisel_lemire_test(2166568064000000238u, -9, 0x10246690000000, 1054);
  eisel_lemire_test(2794967654709307187u, 1, 0x183e132bc608c8, 1087);
  eisel_lemire_test(2794967654709307188u, 1, 0x183e132bc608c9, 1087);
}

// Check the fallback states for the algorithm:
TEST_F(LlvmLibcStrToDblTest, EiselLemireFallbackStates) {
  // This number can't be evaluated by Eisel-Lemire since it's exactly 1024 away
  // from both of its closest floating point approximations
  // (12345678901234548736 and 12345678901234550784)
  ASSERT_FALSE(
      internal::eisel_lemire<double>({12345678901234549760u, 0}).has_value());
}

TEST_F(LlvmLibcStrToDblTest, SimpleDecimalConversion64BasicWholeNumbers) {
  simple_decimal_conversion_test("123456789012345678900", 0x1AC53A7E04BCDA,
                                 1089);
  simple_decimal_conversion_test("123", 0x1EC00000000000, 1029);
  simple_decimal_conversion_test("12345678901234549760", 0x156A95319D63D8,
                                 1086);
}

TEST_F(LlvmLibcStrToDblTest, SimpleDecimalConversion64BasicDecimals) {
  simple_decimal_conversion_test("1.2345", 0x13c083126e978d, 1023);
  simple_decimal_conversion_test(".2345", 0x1e04189374bc6a, 1020);
  simple_decimal_conversion_test(".299792458", 0x132fccb4aca314, 1021);
}

TEST_F(LlvmLibcStrToDblTest, SimpleDecimalConversion64BasicExponents) {
  simple_decimal_conversion_test("1e10", 0x12a05f20000000, 1056);
  simple_decimal_conversion_test("1e-10", 0x1b7cdfd9d7bdbb, 989);
  simple_decimal_conversion_test("1e300", 0x17e43c8800759c, 2019);
  simple_decimal_conversion_test("1e-300", 0x156e1fc2f8f359, 26);
}

TEST_F(LlvmLibcStrToDblTest, SimpleDecimalConversion64BasicSubnormals) {
  simple_decimal_conversion_test("1e-320", 0x7e8, 0, ERANGE);
  simple_decimal_conversion_test("1e-308", 0x730d67819e8d2, 0, ERANGE);
  simple_decimal_conversion_test("2.9e-308", 0x14da6df5e4bcc8, 1);
}

TEST_F(LlvmLibcStrToDblTest, SimpleDecimalConversion64SubnormalRounding) {

  // Technically you can keep adding digits until you hit the truncation limit,
  // but this is the shortest string that results in the maximum subnormal that
  // I found.
  simple_decimal_conversion_test("2.225073858507201e-308", 0xfffffffffffff, 0,
                                 ERANGE);

  // Same here, if you were to extend the max subnormal out for another 800
  // digits, incrementing any one of those digits would create a normal number.
  simple_decimal_conversion_test("2.2250738585072012e-308", 0x10000000000000,
                                 1);
}

TEST(LlvmLibcStrToDblTest, SimpleDecimalConversionExtraTypes) {
  uint64_t double_output_mantissa = 0;
  uint32_t output_exp2 = 0;

  auto double_result =
      internal::simple_decimal_conversion<double>("123456789012345678900");

  double_output_mantissa = double_result.num.mantissa;
  output_exp2 = static_cast<uint32_t>(double_result.num.exponent);

  EXPECT_EQ(double_output_mantissa, uint64_t(0x1AC53A7E04BCDA));
  EXPECT_EQ(output_exp2, uint32_t(1089));
  EXPECT_EQ(double_result.error, 0);
}

} // namespace LIBC_NAMESPACE_DECL
