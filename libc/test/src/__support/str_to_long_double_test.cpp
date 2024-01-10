#include "str_to_fp_test.h"

namespace LIBC_NAMESPACE {

using LlvmLibcStrToLongDblTest = LlvmLibcStrToFloatTest<long double>;

#if defined(LIBC_LONG_DOUBLE_IS_FLOAT64)

TEST_F(LlvmLibcStrToLongDblTest, EiselLemireFloat64AsLongDouble) {
  eisel_lemire_test(123, 0, 0x1EC00000000000, 1029);
}

#elif defined(LIBC_LONG_DOUBLE_IS_X86_FLOAT80)

TEST_F(LlvmLibcStrToLongDblTest, EiselLemireFloat80Simple) {
  eisel_lemire_test(123, 0, 0xf600000000000000, 16389);
  eisel_lemire_test(12345678901234568192u, 0, 0xab54a98ceb1f0c00, 16446);
}

TEST_F(LlvmLibcStrToLongDblTest, EiselLemireFloat80LongerMantissa) {
  eisel_lemire_test((UInt128(0x1234567812345678) << 64) +
                        UInt128(0x1234567812345678),
                    0, 0x91a2b3c091a2b3c1, 16507);
  eisel_lemire_test((UInt128(0x1234567812345678) << 64) +
                        UInt128(0x1234567812345678),
                    300, 0xd97757de56adb65c, 17503);
  eisel_lemire_test((UInt128(0x1234567812345678) << 64) +
                        UInt128(0x1234567812345678),
                    -300, 0xc30feb9a7618457d, 15510);
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

#else // Quad precision long double

TEST_F(LlvmLibcStrToLongDblTest, EiselLemireFloat128Simple) {
  eisel_lemire_test(123, 0, (UInt128(0x1ec0000000000) << 64), 16389);
  eisel_lemire_test(
      12345678901234568192u, 0,
      (UInt128(0x156a95319d63e) << 64) + UInt128(0x1800000000000000), 16446);
}

TEST_F(LlvmLibcStrToLongDblTest, EiselLemireFloat128LongerMantissa) {
  eisel_lemire_test(
      (UInt128(0x1234567812345678) << 64) + UInt128(0x1234567812345678), 0,
      (UInt128(0x1234567812345) << 64) + UInt128(0x6781234567812345), 16507);
  eisel_lemire_test(
      (UInt128(0x1234567812345678) << 64) + UInt128(0x1234567812345678), 300,
      (UInt128(0x1b2eeafbcad5b) << 64) + UInt128(0x6cb8b4451dfcde19), 17503);
  eisel_lemire_test(
      (UInt128(0x1234567812345678) << 64) + UInt128(0x1234567812345678), -300,
      (UInt128(0x1861fd734ec30) << 64) + UInt128(0x8afa7189f0f7595f), 15510);
}

TEST_F(LlvmLibcStrToLongDblTest, EiselLemireFloat128Fallback) {
  ASSERT_FALSE(
      internal::eisel_lemire<long double>(
          {(UInt128(0x5ce0e9a56015fec5) << 64) + UInt128(0xaadfa328ae39b333),
           1})
          .has_value());
}

#endif

} // namespace LIBC_NAMESPACE
