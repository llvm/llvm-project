//===-- Unittests for str_to_float ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/UInt128.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/str_to_float.h"

#include "utils/UnitTest/Test.h"

class LlvmLibcStrToFloatTest : public __llvm_libc::testing::Test {
public:
  template <class T>
  void clinger_fast_path_test(
      const typename __llvm_libc::fputil::FPBits<T>::UIntType inputMantissa,
      const int32_t inputExp10,
      const typename __llvm_libc::fputil::FPBits<T>::UIntType
          expectedOutputMantissa,
      const uint32_t expectedOutputExp2) {
    typename __llvm_libc::fputil::FPBits<T>::UIntType actual_output_mantissa =
        0;
    uint32_t actual_output_exp2 = 0;

    ASSERT_TRUE(__llvm_libc::internal::clinger_fast_path<T>(
        inputMantissa, inputExp10, &actual_output_mantissa,
        &actual_output_exp2));
    EXPECT_EQ(actual_output_mantissa, expectedOutputMantissa);
    EXPECT_EQ(actual_output_exp2, expectedOutputExp2);
  }

  template <class T>
  void clinger_fast_path_fails_test(
      const typename __llvm_libc::fputil::FPBits<T>::UIntType inputMantissa,
      const int32_t inputExp10) {
    typename __llvm_libc::fputil::FPBits<T>::UIntType actual_output_mantissa =
        0;
    uint32_t actual_output_exp2 = 0;

    ASSERT_FALSE(__llvm_libc::internal::clinger_fast_path<T>(
        inputMantissa, inputExp10, &actual_output_mantissa,
        &actual_output_exp2));
  }

  template <class T>
  void eisel_lemire_test(
      const typename __llvm_libc::fputil::FPBits<T>::UIntType inputMantissa,
      const int32_t inputExp10,
      const typename __llvm_libc::fputil::FPBits<T>::UIntType
          expectedOutputMantissa,
      const uint32_t expectedOutputExp2) {
    typename __llvm_libc::fputil::FPBits<T>::UIntType actual_output_mantissa =
        0;
    uint32_t actual_output_exp2 = 0;

    ASSERT_TRUE(__llvm_libc::internal::eisel_lemire<T>(
        inputMantissa, inputExp10, &actual_output_mantissa,
        &actual_output_exp2));
    EXPECT_EQ(actual_output_mantissa, expectedOutputMantissa);
    EXPECT_EQ(actual_output_exp2, expectedOutputExp2);
  }

  template <class T>
  void simple_decimal_conversion_test(
      const char *__restrict numStart,
      const typename __llvm_libc::fputil::FPBits<T>::UIntType
          expectedOutputMantissa,
      const uint32_t expectedOutputExp2, const int expectedErrno = 0) {
    typename __llvm_libc::fputil::FPBits<T>::UIntType actual_output_mantissa =
        0;
    uint32_t actual_output_exp2 = 0;
    errno = 0;

    __llvm_libc::internal::simple_decimal_conversion<T>(
        numStart, &actual_output_mantissa, &actual_output_exp2);
    EXPECT_EQ(actual_output_mantissa, expectedOutputMantissa);
    EXPECT_EQ(actual_output_exp2, expectedOutputExp2);
    EXPECT_EQ(errno, expectedErrno);
  }
};

TEST(LlvmLibcStrToFloatTest, LeadingZeroes) {
  uint64_t test_num64 = 1;
  uint32_t num_of_zeroes = 63;
  EXPECT_EQ(__llvm_libc::internal::leading_zeroes<uint64_t>(0), 64u);
  for (; num_of_zeroes < 64; test_num64 <<= 1, num_of_zeroes--) {
    EXPECT_EQ(__llvm_libc::internal::leading_zeroes<uint64_t>(test_num64),
              num_of_zeroes);
  }

  test_num64 = 3;
  num_of_zeroes = 62;
  for (; num_of_zeroes > 63; test_num64 <<= 1, num_of_zeroes--) {
    EXPECT_EQ(__llvm_libc::internal::leading_zeroes<uint64_t>(test_num64),
              num_of_zeroes);
  }

  EXPECT_EQ(__llvm_libc::internal::leading_zeroes<uint64_t>(0xffffffffffffffff),
            0u);

  test_num64 = 1;
  num_of_zeroes = 63;
  for (; num_of_zeroes > 63;
       test_num64 = (test_num64 << 1) + 1, num_of_zeroes--) {
    EXPECT_EQ(__llvm_libc::internal::leading_zeroes<uint64_t>(test_num64),
              num_of_zeroes);
  }

  uint64_t test_num32 = 1;
  num_of_zeroes = 31;
  EXPECT_EQ(__llvm_libc::internal::leading_zeroes<uint32_t>(0), 32u);
  for (; num_of_zeroes < 32; test_num32 <<= 1, num_of_zeroes--) {
    EXPECT_EQ(__llvm_libc::internal::leading_zeroes<uint32_t>(test_num32),
              num_of_zeroes);
  }

  EXPECT_EQ(__llvm_libc::internal::leading_zeroes<uint32_t>(0xffffffff), 0u);
}

TEST_F(LlvmLibcStrToFloatTest, ClingerFastPathFloat64Simple) {
  clinger_fast_path_test<double>(123, 0, 0xEC00000000000, 1029);
  clinger_fast_path_test<double>(1234567890123456, 1, 0x5ee2a2eb5a5c0, 1076);
  clinger_fast_path_test<double>(1234567890, -10, 0xf9add3739635f, 1019);
}

TEST_F(LlvmLibcStrToFloatTest, ClingerFastPathFloat64ExtendedExp) {
  clinger_fast_path_test<double>(1, 30, 0x93e5939a08cea, 1122);
  clinger_fast_path_test<double>(1, 37, 0xe17b84357691b, 1145);
  clinger_fast_path_fails_test<double>(10, 37);
  clinger_fast_path_fails_test<double>(1, 100);
}

TEST_F(LlvmLibcStrToFloatTest, ClingerFastPathFloat64NegativeExp) {
  clinger_fast_path_test<double>(1, -10, 0xb7cdfd9d7bdbb, 989);
  clinger_fast_path_test<double>(1, -20, 0x79ca10c924223, 956);
  clinger_fast_path_fails_test<double>(1, -25);
}

TEST_F(LlvmLibcStrToFloatTest, ClingerFastPathFloat32Simple) {
  clinger_fast_path_test<float>(123, 0, 0x760000, 133);
  clinger_fast_path_test<float>(1234567, 1, 0x3c6146, 150);
  clinger_fast_path_test<float>(12345, -5, 0x7cd35b, 123);
}

TEST_F(LlvmLibcStrToFloatTest, ClingerFastPathFloat32ExtendedExp) {
  clinger_fast_path_test<float>(1, 15, 0x635fa9, 176);
  clinger_fast_path_test<float>(1, 17, 0x31a2bc, 183);
  clinger_fast_path_fails_test<float>(10, 17);
  clinger_fast_path_fails_test<float>(1, 50);
}

TEST_F(LlvmLibcStrToFloatTest, ClingerFastPathFloat32NegativeExp) {
  clinger_fast_path_test<float>(1, -5, 0x27c5ac, 110);
  clinger_fast_path_test<float>(1, -10, 0x5be6ff, 93);
  clinger_fast_path_fails_test<float>(1, -15);
}

TEST_F(LlvmLibcStrToFloatTest, EiselLemireFloat64Simple) {
  eisel_lemire_test<double>(12345678901234567890u, 1, 0x1AC53A7E04BCDA, 1089);
  eisel_lemire_test<double>(123, 0, 0x1EC00000000000, 1029);
  eisel_lemire_test<double>(12345678901234568192u, 0, 0x156A95319D63E2, 1086);
}

TEST_F(LlvmLibcStrToFloatTest, EiselLemireFloat64SpecificFailures) {
  // These test cases have caused failures in the past.
  eisel_lemire_test<double>(358416272, -33, 0x1BBB2A68C9D0B9, 941);
  eisel_lemire_test<double>(2166568064000000238u, -9, 0x10246690000000, 1054);
  eisel_lemire_test<double>(2794967654709307187u, 1, 0x183e132bc608c8, 1087);
  eisel_lemire_test<double>(2794967654709307188u, 1, 0x183e132bc608c9, 1087);
}

TEST_F(LlvmLibcStrToFloatTest, EiselLemireFallbackStates) {
  // Check the fallback states for the algorithm:
  uint32_t float_output_mantissa = 0;
  uint64_t double_output_mantissa = 0;
  uint32_t output_exp2 = 0;

  // This number can't be evaluated by Eisel-Lemire since it's exactly 1024 away
  // from both of its closest floating point approximations
  // (12345678901234548736 and 12345678901234550784)
  ASSERT_FALSE(__llvm_libc::internal::eisel_lemire<double>(
      12345678901234549760u, 0, &double_output_mantissa, &output_exp2));

  ASSERT_FALSE(__llvm_libc::internal::eisel_lemire<float>(
      20040229, 0, &float_output_mantissa, &output_exp2));
}

TEST_F(LlvmLibcStrToFloatTest, SimpleDecimalConversion64BasicWholeNumbers) {
  simple_decimal_conversion_test<double>("123456789012345678900",
                                         0x1AC53A7E04BCDA, 1089);
  simple_decimal_conversion_test<double>("123", 0x1EC00000000000, 1029);
  simple_decimal_conversion_test<double>("12345678901234549760",
                                         0x156A95319D63D8, 1086);
}

TEST_F(LlvmLibcStrToFloatTest, SimpleDecimalConversion64BasicDecimals) {
  simple_decimal_conversion_test<double>("1.2345", 0x13c083126e978d, 1023);
  simple_decimal_conversion_test<double>(".2345", 0x1e04189374bc6a, 1020);
  simple_decimal_conversion_test<double>(".299792458", 0x132fccb4aca314, 1021);
}

TEST_F(LlvmLibcStrToFloatTest, SimpleDecimalConversion64BasicExponents) {
  simple_decimal_conversion_test<double>("1e10", 0x12a05f20000000, 1056);
  simple_decimal_conversion_test<double>("1e-10", 0x1b7cdfd9d7bdbb, 989);
  simple_decimal_conversion_test<double>("1e300", 0x17e43c8800759c, 2019);
  simple_decimal_conversion_test<double>("1e-300", 0x156e1fc2f8f359, 26);
}

TEST_F(LlvmLibcStrToFloatTest, SimpleDecimalConversion64BasicSubnormals) {
  simple_decimal_conversion_test<double>("1e-320", 0x7e8, 0, ERANGE);
  simple_decimal_conversion_test<double>("1e-308", 0x730d67819e8d2, 0, ERANGE);
  simple_decimal_conversion_test<double>("2.9e-308", 0x14da6df5e4bcc8, 1);
}

TEST_F(LlvmLibcStrToFloatTest, SimpleDecimalConversion64SubnormalRounding) {

  // Technically you can keep adding digits until you hit the truncation limit,
  // but this is the shortest string that results in the maximum subnormal that
  // I found.
  simple_decimal_conversion_test<double>("2.225073858507201e-308",
                                         0xfffffffffffff, 0, ERANGE);

  // Same here, if you were to extend the max subnormal out for another 800
  // digits, incrementing any one of those digits would create a normal number.
  simple_decimal_conversion_test<double>("2.2250738585072012e-308",
                                         0x10000000000000, 1);
}

TEST_F(LlvmLibcStrToFloatTest, SimpleDecimalConversion32SpecificFailures) {
  simple_decimal_conversion_test<float>(
      "1.4012984643248170709237295832899161312802619418765e-45", 0x1, 0,
      ERANGE);
  simple_decimal_conversion_test<float>(
      "7."
      "006492321624085354618647916449580656401309709382578858785341419448955413"
      "42930300743319094181060791015625e-46",
      0x0, 0, ERANGE);
}

TEST(LlvmLibcStrToFloatTest, SimpleDecimalConversionExtraTypes) {
  uint32_t float_output_mantissa = 0;
  uint32_t output_exp2 = 0;

  errno = 0;
  __llvm_libc::internal::simple_decimal_conversion<float>(
      "123456789012345678900", &float_output_mantissa, &output_exp2);
  EXPECT_EQ(float_output_mantissa, uint32_t(0xd629d4));
  EXPECT_EQ(output_exp2, uint32_t(193));
  EXPECT_EQ(errno, 0);

  uint64_t double_output_mantissa = 0;
  output_exp2 = 0;

  errno = 0;
  __llvm_libc::internal::simple_decimal_conversion<double>(
      "123456789012345678900", &double_output_mantissa, &output_exp2);
  EXPECT_EQ(double_output_mantissa, uint64_t(0x1AC53A7E04BCDA));
  EXPECT_EQ(output_exp2, uint32_t(1089));
  EXPECT_EQ(errno, 0);
}

#if defined(LONG_DOUBLE_IS_DOUBLE)
TEST_F(LlvmLibcStrToFloatTest, EiselLemireFloat64AsLongDouble) {
  eisel_lemire_test<long double>(123, 0, 0x1EC00000000000, 1029);
}
#elif defined(SPECIAL_X86_LONG_DOUBLE)
TEST_F(LlvmLibcStrToFloatTest, EiselLemireFloat80Simple) {
  eisel_lemire_test<long double>(123, 0, 0xf600000000000000, 16389);
  eisel_lemire_test<long double>(12345678901234568192u, 0, 0xab54a98ceb1f0c00,
                                 16446);
}

TEST_F(LlvmLibcStrToFloatTest, EiselLemireFloat80LongerMantissa) {
  eisel_lemire_test<long double>((UInt128(0x1234567812345678) << 64) +
                                     UInt128(0x1234567812345678),
                                 0, 0x91a2b3c091a2b3c1, 16507);
  eisel_lemire_test<long double>((UInt128(0x1234567812345678) << 64) +
                                     UInt128(0x1234567812345678),
                                 300, 0xd97757de56adb65c, 17503);
  eisel_lemire_test<long double>((UInt128(0x1234567812345678) << 64) +
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
TEST_F(LlvmLibcStrToFloatTest, EiselLemireFloat80TableLimits) {
  eisel_lemire_test<long double>(1, 347, 0xd13eb46469447567, 17535);
  eisel_lemire_test<long double>(1, -348, 0xfa8fd5a0081c0288, 15226);
}

TEST_F(LlvmLibcStrToFloatTest, EiselLemireFloat80Fallback) {
  uint32_t outputExp2 = 0;
  UInt128 quadOutputMantissa = 0;

  // This number is halfway between two possible results, and the algorithm
  // can't determine which is correct.
  ASSERT_FALSE(__llvm_libc::internal::eisel_lemire<long double>(
      12345678901234567890u, 1, &quadOutputMantissa, &outputExp2));

  // These numbers' exponents are out of range for the current powers of ten
  // table.
  ASSERT_FALSE(__llvm_libc::internal::eisel_lemire<long double>(
      1, 1000, &quadOutputMantissa, &outputExp2));
  ASSERT_FALSE(__llvm_libc::internal::eisel_lemire<long double>(
      1, -1000, &quadOutputMantissa, &outputExp2));
}
#else // Quad precision long double
TEST_F(LlvmLibcStrToFloatTest, EiselLemireFloat128Simple) {
  eisel_lemire_test<long double>(123, 0, (UInt128(0x1ec0000000000) << 64),
                                 16389);
  eisel_lemire_test<long double>(
      12345678901234568192u, 0,
      (UInt128(0x156a95319d63e) << 64) + UInt128(0x1800000000000000), 16446);
}

TEST_F(LlvmLibcStrToFloatTest, EiselLemireFloat128LongerMantissa) {
  eisel_lemire_test<long double>(
      (UInt128(0x1234567812345678) << 64) + UInt128(0x1234567812345678), 0,
      (UInt128(0x1234567812345) << 64) + UInt128(0x6781234567812345), 16507);
  eisel_lemire_test<long double>(
      (UInt128(0x1234567812345678) << 64) + UInt128(0x1234567812345678), 300,
      (UInt128(0x1b2eeafbcad5b) << 64) + UInt128(0x6cb8b4451dfcde19), 17503);
  eisel_lemire_test<long double>(
      (UInt128(0x1234567812345678) << 64) + UInt128(0x1234567812345678), -300,
      (UInt128(0x1861fd734ec30) << 64) + UInt128(0x8afa7189f0f7595f), 15510);
}

TEST_F(LlvmLibcStrToFloatTest, EiselLemireFloat128Fallback) {
  uint32_t outputExp2 = 0;
  UInt128 quadOutputMantissa = 0;

  ASSERT_FALSE(__llvm_libc::internal::eisel_lemire<long double>(
      (UInt128(0x5ce0e9a56015fec5) << 64) + UInt128(0xaadfa328ae39b333), 1,
      &quadOutputMantissa, &outputExp2));
}
#endif
