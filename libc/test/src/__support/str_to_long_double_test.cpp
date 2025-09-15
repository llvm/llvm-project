//===-- Unittests for str_to_float<long double> ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/macros/config.h"
#include "str_to_fp_test.h"

#include "src/__support/integer_literals.h"

namespace LIBC_NAMESPACE_DECL {

using LlvmLibcStrToLongDblTest = LlvmLibcStrToFloatTest<long double>;
using LIBC_NAMESPACE::operator""_u128;

#if defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT64)

TEST_F(LlvmLibcStrToLongDblTest, EiselLemireFloat64AsLongDouble) {
  eisel_lemire_test(123, 0, 0x1EC00000000000, 1029);
}

#elif defined(LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80)

TEST_F(LlvmLibcStrToLongDblTest, EiselLemireFloat80Simple) {
  eisel_lemire_test(123, 0, 0xf600000000000000, 16389);
  eisel_lemire_test(12345678901234568192u, 0, 0xab54a98ceb1f0c00, 16446);
}

TEST_F(LlvmLibcStrToLongDblTest, EiselLemireFloat80LongerMantissa) {
#if __SIZEOF_LONG_DOUBLE__ == 16
  eisel_lemire_test(0x12345678'12345678'12345678'12345678_u128, 0,
                    0x91a2b3c091a2b3c1, 16507);
  eisel_lemire_test(0x12345678'12345678'12345678'12345678_u128, 300,
                    0xd97757de56adb65c, 17503);
  eisel_lemire_test(0x12345678'12345678'12345678'12345678_u128, -300,
                    0xc30feb9a7618457d, 15510);
#elif __SIZEOF_LONG_DOUBLE__ == 12
  eisel_lemire_test(0x12345678'12345678'12345678_u96, 0, 0x91a2b3c091a2b3c1,
                    16475);
  eisel_lemire_test(0x12345678'12345678'12345678_u96, 300, 0xd97757de56adb65c,
                    17471);
  eisel_lemire_test(0x12345678'12345678'12345678_u96, -300, 0xc30feb9a7618457d,
                    15478);
#else
#error "unhandled long double type"
#endif
}

// These tests check numbers at the edge of the DETAILED_POWERS_OF_TEN table.
// This doesn't reach very far into the range for long doubles, since it's sized
// for doubles and their 11 exponent bits, and not for long doubles and their
// 15 exponent bits. This is a known tradeoff, and was made because a proper
// long double table would be approximately 16 times longer (specifically the
// maximum exponent would need to be about 5000, leading to a 10,000 entry
// table). This would have significant memory and storage costs all the time to
// speed up a relatively uncommon path.
TEST_F(LlvmLibcStrToLongDblTest, EiselLemireFloat80TableLimits) {
  eisel_lemire_test(1, 347, 0xd13eb46469447567, 17535);
  eisel_lemire_test(1, -348, 0xfa8fd5a0081c0288, 15226);
}

TEST_F(LlvmLibcStrToLongDblTest, EiselLemireFloat80Fallback) {
  // This number is halfway between two possible results, and the algorithm
  // can't determine which is correct.
  ASSERT_FALSE(internal::eisel_lemire<long double>({12345678901234567890u, 1})
                   .has_value());

  // These numbers' exponents are out of range for the current powers of ten
  // table.
  ASSERT_FALSE(internal::eisel_lemire<long double>({1, 1000}).has_value());
  ASSERT_FALSE(internal::eisel_lemire<long double>({1, -1000}).has_value());
}

TEST_F(LlvmLibcStrToLongDblTest, ClingerFastPathFloat80Simple) {
  clinger_fast_path_test(123, 0, 0xf600000000000000, 16389);
  clinger_fast_path_test(1234567, 1, 0xbc61460000000000, 16406);
  clinger_fast_path_test(12345, -5, 0xfcd35a858793dd98, 16379);
}

#elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT128)

TEST_F(LlvmLibcStrToLongDblTest, EiselLemireFloat128Simple) {
  eisel_lemire_test(123, 0, 0x1ec00'00000000'00000000'00000000_u128, 16389);
  eisel_lemire_test(12345678901234568192u, 0,
                    0x156a9'5319d63e'18000000'00000000_u128, 16446);
}

TEST_F(LlvmLibcStrToLongDblTest, EiselLemireFloat128LongerMantissa) {
  eisel_lemire_test(0x12345678'12345678'12345678'12345678_u128, 0,
                    0x12345'67812345'67812345'67812345_u128, 16507);
  eisel_lemire_test(0x12345678'12345678'12345678'12345678_u128, 300,
                    0x1b2ee'afbcad5b'6cb8b445'1dfcde19_u128, 17503);
  eisel_lemire_test(0x12345678'12345678'12345678'12345678_u128, -300,
                    0x1861f'd734ec30'8afa7189'f0f7595f_u128, 15510);
}

TEST_F(LlvmLibcStrToLongDblTest, EiselLemireFloat128Fallback) {
  ASSERT_FALSE(internal::eisel_lemire<long double>(
                   {0x5ce0e9a5'6015fec5'aadfa328'ae39b333_u128, 1})
                   .has_value());
}

TEST_F(LlvmLibcStrToLongDblTest, ClingerFastPathFloat128Simple) {
  clinger_fast_path_test(123, 0, 0x1ec00'00000000'00000000'00000000_u128,
                         16389);
  clinger_fast_path_test(1234567, 1, 0x178c2'8c000000'00000000'00000000_u128,
                         16406);
  clinger_fast_path_test(12345, -5, 0x1f9a6'b50b0f27'bb2fec56'd5cfaace_u128,
                         16379);
}

#else
#error "Unknown long double type"
#endif

} // namespace LIBC_NAMESPACE_DECL
