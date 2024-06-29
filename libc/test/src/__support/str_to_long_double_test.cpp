#include "str_to_fp_test.h"

#include "src/__support/integer_literals.h"

namespace LIBC_NAMESPACE {

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
  eisel_lemire_test(0x12345678'12345678'12345678'12345678_u128, 0,
                    0x91a2b3c091a2b3c1, 16507);
  eisel_lemire_test(0x12345678'12345678'12345678'12345678_u128, 300,
                    0xd97757de56adb65c, 17503);
  eisel_lemire_test(0x12345678'12345678'12345678'12345678_u128, -300,
                    0xc30feb9a7618457d, 15510);
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

#else
#error "Unknown long double type"
#endif

} // namespace LIBC_NAMESPACE
