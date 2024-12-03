//===-- Unittests for str_to_float<float> ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/macros/config.h"
#include "str_to_fp_test.h"

namespace LIBC_NAMESPACE_DECL {

using LlvmLibcStrToFltTest = LlvmLibcStrToFloatTest<float>;

TEST_F(LlvmLibcStrToFltTest, ClingerFastPathFloat32Simple) {
  clinger_fast_path_test(123, 0, 0xf60000, 133);
  clinger_fast_path_test(1234567, 1, 0xbc6146, 150);
  clinger_fast_path_test(12345, -5, 0xfcd35b, 123);
}

TEST_F(LlvmLibcStrToFltTest, ClingerFastPathFloat32ExtendedExp) {
  clinger_fast_path_test(1, 15, 0xe35fa9, 176);
  clinger_fast_path_test(1, 17, 0xb1a2bc, 183);
  clinger_fast_path_fails_test(10, 17);
  clinger_fast_path_fails_test(1, 50);
}

TEST_F(LlvmLibcStrToFltTest, ClingerFastPathFloat32NegativeExp) {
  clinger_fast_path_test(1, -5, 0xa7c5ac, 110);
  clinger_fast_path_test(1, -10, 0xdbe6ff, 93);
  clinger_fast_path_fails_test(1, -15);
}

// Check the fallback states for the algorithm:
TEST_F(LlvmLibcStrToFltTest, EiselLemireFallbackStates) {
  // This number can't be evaluated by Eisel-Lemire since it's exactly 1024 away
  // from both of its closest floating point approximations
  // (12345678901234548736 and 12345678901234550784)
  ASSERT_FALSE(internal::eisel_lemire<float>({20040229, 0}).has_value());
}

TEST_F(LlvmLibcStrToFltTest, SimpleDecimalConversion32SpecificFailures) {
  simple_decimal_conversion_test(
      "1.4012984643248170709237295832899161312802619418765e-45", 0x1, 0,
      ERANGE);
  simple_decimal_conversion_test(
      "7."
      "006492321624085354618647916449580656401309709382578858785341419448955413"
      "42930300743319094181060791015625e-46",
      0x0, 0, ERANGE);
}

TEST(LlvmLibcStrToFltTest, SimpleDecimalConversionExtraTypes) {
  uint32_t float_output_mantissa = 0;
  uint32_t output_exp2 = 0;

  LIBC_NAMESPACE::libc_errno = 0;
  auto float_result =
      internal::simple_decimal_conversion<float>("123456789012345678900");
  float_output_mantissa = float_result.num.mantissa;
  output_exp2 = float_result.num.exponent;
  EXPECT_EQ(float_output_mantissa, uint32_t(0xd629d4));
  EXPECT_EQ(output_exp2, uint32_t(193));
  EXPECT_EQ(float_result.error, 0);
}

} // namespace LIBC_NAMESPACE_DECL
