//===-- Unittests for str_to_float ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/str_to_float.h"

#include "utils/UnitTest/Test.h"

class LlvmLibcStrToFloatTest : public __llvm_libc::testing::Test {
public:
  template <class T>
  void EiselLemireTest(
      const typename __llvm_libc::fputil::FPBits<T>::UIntType inputMantissa,
      const int32_t inputExp10,
      const typename __llvm_libc::fputil::FPBits<T>::UIntType
          expectedOutputMantissa,
      const uint32_t expectedOutputExp2) {
    typename __llvm_libc::fputil::FPBits<T>::UIntType actualOutputMantissa = 0;
    uint32_t actualOutputExp2 = 0;

    ASSERT_TRUE(__llvm_libc::internal::eiselLemire<T>(
        inputMantissa, inputExp10, &actualOutputMantissa, &actualOutputExp2));
    EXPECT_EQ(actualOutputMantissa, expectedOutputMantissa);
    EXPECT_EQ(actualOutputExp2, expectedOutputExp2);
  }

  template <class T>
  void SimpleDecimalConversionTest(
      const char *__restrict numStart,
      const typename __llvm_libc::fputil::FPBits<T>::UIntType
          expectedOutputMantissa,
      const uint32_t expectedOutputExp2, const int expectedErrno = 0) {
    typename __llvm_libc::fputil::FPBits<T>::UIntType actualOutputMantissa = 0;
    uint32_t actualOutputExp2 = 0;
    errno = 0;

    __llvm_libc::internal::simpleDecimalConversion<T>(
        numStart, &actualOutputMantissa, &actualOutputExp2);
    EXPECT_EQ(actualOutputMantissa, expectedOutputMantissa);
    EXPECT_EQ(actualOutputExp2, expectedOutputExp2);
    EXPECT_EQ(errno, expectedErrno);
  }
};

TEST(LlvmLibcStrToFloatTest, LeadingZeroes) {
  uint64_t testNum64 = 1;
  uint32_t numOfZeroes = 63;
  EXPECT_EQ(__llvm_libc::internal::leadingZeroes<uint64_t>(0), 64u);
  for (; numOfZeroes < 64; testNum64 <<= 1, numOfZeroes--) {
    EXPECT_EQ(__llvm_libc::internal::leadingZeroes<uint64_t>(testNum64),
              numOfZeroes);
  }

  testNum64 = 3;
  numOfZeroes = 62;
  for (; numOfZeroes > 63; testNum64 <<= 1, numOfZeroes--) {
    EXPECT_EQ(__llvm_libc::internal::leadingZeroes<uint64_t>(testNum64),
              numOfZeroes);
  }

  EXPECT_EQ(__llvm_libc::internal::leadingZeroes<uint64_t>(0xffffffffffffffff),
            0u);

  testNum64 = 1;
  numOfZeroes = 63;
  for (; numOfZeroes > 63; testNum64 = (testNum64 << 1) + 1, numOfZeroes--) {
    EXPECT_EQ(__llvm_libc::internal::leadingZeroes<uint64_t>(testNum64),
              numOfZeroes);
  }

  uint64_t testNum32 = 1;
  numOfZeroes = 31;
  EXPECT_EQ(__llvm_libc::internal::leadingZeroes<uint32_t>(0), 32u);
  for (; numOfZeroes < 32; testNum32 <<= 1, numOfZeroes--) {
    EXPECT_EQ(__llvm_libc::internal::leadingZeroes<uint32_t>(testNum32),
              numOfZeroes);
  }

  EXPECT_EQ(__llvm_libc::internal::leadingZeroes<uint32_t>(0xffffffff), 0u);
}

TEST_F(LlvmLibcStrToFloatTest, EiselLemireFloat64Simple) {
  EiselLemireTest<double>(12345678901234567890u, 1, 0x1AC53A7E04BCDA, 1089);
  EiselLemireTest<double>(123, 0, 0x1EC00000000000, 1029);
  EiselLemireTest<double>(12345678901234568192u, 0, 0x156A95319D63E2, 1086);
}

TEST_F(LlvmLibcStrToFloatTest, EiselLemireFloat64SpecificFailures) {
  // These test cases have caused failures in the past.
  EiselLemireTest<double>(358416272, -33, 0x1BBB2A68C9D0B9, 941);
  EiselLemireTest<double>(2166568064000000238u, -9, 0x10246690000000, 1054);
  EiselLemireTest<double>(2794967654709307187u, 1, 0x183e132bc608c8, 1087);
  EiselLemireTest<double>(2794967654709307188u, 1, 0x183e132bc608c9, 1087);
}

TEST_F(LlvmLibcStrToFloatTest, EiselLemireFallbackStates) {
  // Check the fallback states for the algorithm:
  uint32_t floatOutputMantissa = 0;
  uint64_t doubleOutputMantissa = 0;
  __uint128_t tooLongMantissa = 0;
  uint32_t outputExp2 = 0;

  // This Eisel-Lemire implementation doesn't support long doubles yet.
  ASSERT_FALSE(__llvm_libc::internal::eiselLemire<long double>(
      tooLongMantissa, 0, &tooLongMantissa, &outputExp2));

  // This number can't be evaluated by Eisel-Lemire since it's exactly 1024 away
  // from both of its closest floating point approximations
  // (12345678901234548736 and 12345678901234550784)
  ASSERT_FALSE(__llvm_libc::internal::eiselLemire<double>(
      12345678901234549760u, 0, &doubleOutputMantissa, &outputExp2));

  ASSERT_FALSE(__llvm_libc::internal::eiselLemire<float>(
      20040229, 0, &floatOutputMantissa, &outputExp2));
}

TEST_F(LlvmLibcStrToFloatTest, SimpleDecimalConversion64BasicWholeNumbers) {
  SimpleDecimalConversionTest<double>("123456789012345678900", 0x1AC53A7E04BCDA,
                                      1089);
  SimpleDecimalConversionTest<double>("123", 0x1EC00000000000, 1029);
  SimpleDecimalConversionTest<double>("12345678901234549760", 0x156A95319D63D8,
                                      1086);
}

TEST_F(LlvmLibcStrToFloatTest, SimpleDecimalConversion64BasicDecimals) {
  SimpleDecimalConversionTest<double>("1.2345", 0x13c083126e978d, 1023);
  SimpleDecimalConversionTest<double>(".2345", 0x1e04189374bc6a, 1020);
  SimpleDecimalConversionTest<double>(".299792458", 0x132fccb4aca314, 1021);
}

TEST_F(LlvmLibcStrToFloatTest, SimpleDecimalConversion64BasicExponents) {
  SimpleDecimalConversionTest<double>("1e10", 0x12a05f20000000, 1056);
  SimpleDecimalConversionTest<double>("1e-10", 0x1b7cdfd9d7bdbb, 989);
  SimpleDecimalConversionTest<double>("1e300", 0x17e43c8800759c, 2019);
  SimpleDecimalConversionTest<double>("1e-300", 0x156e1fc2f8f359, 26);
}

TEST_F(LlvmLibcStrToFloatTest, SimpleDecimalConversion64BasicSubnormals) {
  SimpleDecimalConversionTest<double>("1e-320", 0x7e8, 0);
  SimpleDecimalConversionTest<double>("1e-308", 0x730d67819e8d2, 0);
  SimpleDecimalConversionTest<double>("2.9e-308", 0x14da6df5e4bcc8, 1);
}

TEST_F(LlvmLibcStrToFloatTest, SimpleDecimalConversion64SubnormalRounding) {

  // Technically you can keep adding digits until you hit the truncation limit,
  // but this is the shortest string that results in the maximum subnormal that
  // I found.
  SimpleDecimalConversionTest<double>("2.225073858507201e-308", 0xfffffffffffff,
                                      0);

  // Same here, if you were to extend the max subnormal out for another 800
  // digits, incrementing any one of those digits would create a normal number.
  SimpleDecimalConversionTest<double>("2.2250738585072012e-308",
                                      0x10000000000000, 1);
}

TEST_F(LlvmLibcStrToFloatTest, SimpleDecimalConversion32SpecificFailures) {
  SimpleDecimalConversionTest<float>(
      "1.4012984643248170709237295832899161312802619418765e-45", 0x1, 0);
}

TEST(LlvmLibcStrToFloatTest, SimpleDecimalConversionExtraTypes) {
  uint32_t floatOutputMantissa = 0;
  uint32_t outputExp2 = 0;

  errno = 0;
  __llvm_libc::internal::simpleDecimalConversion<float>(
      "123456789012345678900", &floatOutputMantissa, &outputExp2);
  EXPECT_EQ(floatOutputMantissa, uint32_t(0xd629d4));
  EXPECT_EQ(outputExp2, uint32_t(193));
  EXPECT_EQ(errno, 0);

  uint64_t doubleOutputMantissa = 0;
  outputExp2 = 0;

  errno = 0;
  __llvm_libc::internal::simpleDecimalConversion<double>(
      "123456789012345678900", &doubleOutputMantissa, &outputExp2);
  EXPECT_EQ(doubleOutputMantissa, uint64_t(0x1AC53A7E04BCDA));
  EXPECT_EQ(outputExp2, uint32_t(1089));
  EXPECT_EQ(errno, 0);

  // TODO(michaelrj): Get long double support working.

  // __uint128_t longDoubleOutputMantissa = 0;
  // outputExp2 = 0;

  // errno = 0;
  // __llvm_libc::internal::simpleDecimalConversion<long double>(
  //     "123456789012345678900", &longDoubleOutputMantissa, &outputExp2);
  // EXPECT_EQ(longDoubleOutputMantissa, __uint128_t(0x1AC53A7E04BCDA));
  // EXPECT_EQ(outputExp2, uint32_t(1089));
  // EXPECT_EQ(errno, 0);
}
