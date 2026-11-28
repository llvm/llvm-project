//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for str_to_float<fputil::Float128>
///
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/float128.h"
#include "src/__support/integer_literals.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/properties/types.h"
#include "str_to_fp_test.h"

#include <stddef.h>

namespace LIBC_NAMESPACE_DECL {

using LlvmLibcStrToFloat128Test = LlvmLibcStrToFloatTest<fputil::Float128>;
using LIBC_NAMESPACE::operator""_u128;

TEST_F(LlvmLibcStrToFloat128Test, EiselLemireFloat128Simple) {
  eisel_lemire_test(123, 0, 0x1ec00'00000000'00000000'00000000_u128, 16389);
  eisel_lemire_test(12345678901234568192u, 0,
                    0x156a9'5319d63e'18000000'00000000_u128, 16446);
}

TEST_F(LlvmLibcStrToFloat128Test, EiselLemireFloat128LongerMantissa) {
  eisel_lemire_test(0x12345678'12345678'12345678'12345678_u128, 0,
                    0x12345'67812345'67812345'67812345_u128, 16507);
  eisel_lemire_test(0x12345678'12345678'12345678'12345678_u128, 300,
                    0x1b2ee'afbcad5b'6cb8b445'1dfcde19_u128, 17503);
  eisel_lemire_test(0x12345678'12345678'12345678'12345678_u128, -300,
                    0x1861f'd734ec30'8afa7189'f0f7595f_u128, 15510);
}

TEST_F(LlvmLibcStrToFloat128Test, EiselLemireFloat128Fallback) {
  ASSERT_FALSE(internal::eisel_lemire<fputil::Float128>(
                   {0x5ce0e9a5'6015fec5'aadfa328'ae39b333_u128, 1})
                   .has_value());
}

TEST_F(LlvmLibcStrToFloat128Test, ClingerFastPathFloat128Simple) {
  clinger_fast_path_test(123, 0, 0x1ec00'00000000'00000000'00000000_u128,
                         16389);
  clinger_fast_path_test(1234567, 1, 0x178c2'8c000000'00000000'00000000_u128,
                         16406);
  clinger_fast_path_test(12345, -5, 0x1f9a6'b50b0f27'bb2fec56'd5cfaace_u128,
                         16379);
}

#if defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT128)

TEST_F(LlvmLibcStrToFloat128Test, ClingerConstsEqualLongDouble) {
  using F128Consts = internal::ClingerConsts<fputil::Float128>;
  using LDConsts = internal::ClingerConsts<long double>;

  EXPECT_EQ(F128Consts::EXACT_POWERS_OF_TEN, LDConsts::EXACT_POWERS_OF_TEN);
  EXPECT_EQ(F128Consts::DIGITS_IN_MANTISSA, LDConsts::DIGITS_IN_MANTISSA);
  EXPECT_EQ(
      fputil::FPBits<fputil::Float128>(F128Consts::MAX_EXACT_INT).uintval(),
      fputil::FPBits<long double>(LDConsts::MAX_EXACT_INT).uintval());

  constexpr size_t F128_LEN = sizeof(F128Consts::POWERS_OF_TEN_ARRAY) /
                              sizeof(F128Consts::POWERS_OF_TEN_ARRAY[0]);
  constexpr size_t LD_LEN = sizeof(LDConsts::POWERS_OF_TEN_ARRAY) /
                            sizeof(LDConsts::POWERS_OF_TEN_ARRAY[0]);
  ASSERT_EQ(F128_LEN, LD_LEN);
  for (size_t i = 0; i < F128_LEN; ++i) {
    EXPECT_EQ(
        fputil::FPBits<fputil::Float128>(F128Consts::POWERS_OF_TEN_ARRAY[i])
            .uintval(),
        fputil::FPBits<long double>(LDConsts::POWERS_OF_TEN_ARRAY[i])
            .uintval());
  }
}

#endif // defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT128)

} // namespace LIBC_NAMESPACE_DECL
