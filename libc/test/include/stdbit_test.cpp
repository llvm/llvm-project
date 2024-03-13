//===-- Unittests for stdbit ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDSList-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "test/UnitTest/Test.h"

/*
 * The intent of this test is validate that:
 * 1. We provide the definition of the various type generic macros of stdbit.h
 * (the macros are transitively included from stdbit-macros.h by stdbit.h).
 * 2. It dispatches to the correct underlying function.
 * Because unit tests build without public packaging, the object files produced
 * do not contain non-namespaced symbols.
 */

/*
 * Declare these BEFORE including stdbit-macros.h so that this test may still be
 * run even if a given target doesn't yet have these individual entrypoints
 * enabled.
 */
#include "stdbit_stub.h"

#include "include/llvm-libc-macros/stdbit-macros.h"

TEST(LlvmLibcStdbitTest, TypeGenericMacroLeadingZeros) {
  EXPECT_EQ(stdc_leading_zeros(static_cast<unsigned char>(0U)), 0xAAU);
  EXPECT_EQ(stdc_leading_zeros(static_cast<unsigned short>(0U)), 0xABU);
  EXPECT_EQ(stdc_leading_zeros(0U), 0xACU);
  EXPECT_EQ(stdc_leading_zeros(0UL), 0xADU);
  EXPECT_EQ(stdc_leading_zeros(0ULL), 0xAEU);
}

TEST(LlvmLibcStdbitTest, TypeGenericMacroLeadingOnes) {
  EXPECT_EQ(stdc_leading_ones(static_cast<unsigned char>(0U)), 0xBAU);
  EXPECT_EQ(stdc_leading_ones(static_cast<unsigned short>(0U)), 0xBBU);
  EXPECT_EQ(stdc_leading_ones(0U), 0xBCU);
  EXPECT_EQ(stdc_leading_ones(0UL), 0xBDU);
  EXPECT_EQ(stdc_leading_ones(0ULL), 0xBEU);
}

TEST(LlvmLibcStdbitTest, TypeGenericMacroTrailingZeros) {
  EXPECT_EQ(stdc_trailing_zeros(static_cast<unsigned char>(0U)), 0xCAU);
  EXPECT_EQ(stdc_trailing_zeros(static_cast<unsigned short>(0U)), 0xCBU);
  EXPECT_EQ(stdc_trailing_zeros(0U), 0xCCU);
  EXPECT_EQ(stdc_trailing_zeros(0UL), 0xCDU);
  EXPECT_EQ(stdc_trailing_zeros(0ULL), 0xCEU);
}

TEST(LlvmLibcStdbitTest, TypeGenericMacroTrailingOnes) {
  EXPECT_EQ(stdc_trailing_ones(static_cast<unsigned char>(0U)), 0xDAU);
  EXPECT_EQ(stdc_trailing_ones(static_cast<unsigned short>(0U)), 0xDBU);
  EXPECT_EQ(stdc_trailing_ones(0U), 0xDCU);
  EXPECT_EQ(stdc_trailing_ones(0UL), 0xDDU);
  EXPECT_EQ(stdc_trailing_ones(0ULL), 0xDEU);
}

TEST(LlvmLibcStdbitTest, TypeGenericMacroFirstLeadingZero) {
  EXPECT_EQ(stdc_first_leading_zero(static_cast<unsigned char>(0U)), 0xEAU);
  EXPECT_EQ(stdc_first_leading_zero(static_cast<unsigned short>(0U)), 0xEBU);
  EXPECT_EQ(stdc_first_leading_zero(0U), 0xECU);
  EXPECT_EQ(stdc_first_leading_zero(0UL), 0xEDU);
  EXPECT_EQ(stdc_first_leading_zero(0ULL), 0xEEU);
}

TEST(LlvmLibcStdbitTest, TypeGenericMacroFirstLeadingOne) {
  EXPECT_EQ(stdc_first_leading_one(static_cast<unsigned char>(0U)), 0xFAU);
  EXPECT_EQ(stdc_first_leading_one(static_cast<unsigned short>(0U)), 0xFBU);
  EXPECT_EQ(stdc_first_leading_one(0U), 0xFCU);
  EXPECT_EQ(stdc_first_leading_one(0UL), 0xFDU);
  EXPECT_EQ(stdc_first_leading_one(0ULL), 0xFEU);
}

TEST(LlvmLibcStdbitTest, TypeGenericMacroFirstTrailingZero) {
  EXPECT_EQ(stdc_first_trailing_zero(static_cast<unsigned char>(0U)), 0x0AU);
  EXPECT_EQ(stdc_first_trailing_zero(static_cast<unsigned short>(0U)), 0x0BU);
  EXPECT_EQ(stdc_first_trailing_zero(0U), 0x0CU);
  EXPECT_EQ(stdc_first_trailing_zero(0UL), 0x0DU);
  EXPECT_EQ(stdc_first_trailing_zero(0ULL), 0x0EU);
}

TEST(LlvmLibcStdbitTest, TypeGenericMacroFirstTrailingOne) {
  EXPECT_EQ(stdc_first_trailing_one(static_cast<unsigned char>(0U)), 0x1AU);
  EXPECT_EQ(stdc_first_trailing_one(static_cast<unsigned short>(0U)), 0x1BU);
  EXPECT_EQ(stdc_first_trailing_one(0U), 0x1CU);
  EXPECT_EQ(stdc_first_trailing_one(0UL), 0x1DU);
  EXPECT_EQ(stdc_first_trailing_one(0ULL), 0x1EU);
}

TEST(LlvmLibcStdbitTest, TypeGenericMacroCountZeros) {
  EXPECT_EQ(stdc_count_zeros(static_cast<unsigned char>(0U)), 0x2AU);
  EXPECT_EQ(stdc_count_zeros(static_cast<unsigned short>(0U)), 0x2BU);
  EXPECT_EQ(stdc_count_zeros(0U), 0x2CU);
  EXPECT_EQ(stdc_count_zeros(0UL), 0x2DU);
  EXPECT_EQ(stdc_count_zeros(0ULL), 0x2EU);
}

TEST(LlvmLibcStdbitTest, TypeGenericMacroCountOnes) {
  EXPECT_EQ(stdc_count_ones(static_cast<unsigned char>(0U)), 0x3AU);
  EXPECT_EQ(stdc_count_ones(static_cast<unsigned short>(0U)), 0x3BU);
  EXPECT_EQ(stdc_count_ones(0U), 0x3CU);
  EXPECT_EQ(stdc_count_ones(0UL), 0x3DU);
  EXPECT_EQ(stdc_count_ones(0ULL), 0x3EU);
}

TEST(LlvmLibcStdbitTest, TypeGenericMacroHasSingleBit) {
  EXPECT_EQ(stdc_has_single_bit(static_cast<unsigned char>(1U)), false);
  EXPECT_EQ(stdc_has_single_bit(static_cast<unsigned short>(1U)), false);
  EXPECT_EQ(stdc_has_single_bit(1U), false);
  EXPECT_EQ(stdc_has_single_bit(1UL), false);
  EXPECT_EQ(stdc_has_single_bit(1ULL), false);
}

TEST(LlvmLibcStdbitTest, TypeGenericMacroBitWidth) {
  EXPECT_EQ(stdc_bit_width(static_cast<unsigned char>(1U)), 0x4AU);
  EXPECT_EQ(stdc_bit_width(static_cast<unsigned short>(1U)), 0x4BU);
  EXPECT_EQ(stdc_bit_width(1U), 0x4CU);
  EXPECT_EQ(stdc_bit_width(1UL), 0x4DU);
  EXPECT_EQ(stdc_bit_width(1ULL), 0x4EU);
}

TEST(LlvmLibcStdbitTest, TypeGenericMacroBitFloor) {
  EXPECT_EQ(stdc_bit_floor(static_cast<unsigned char>(0U)),
            static_cast<unsigned char>(0x5AU));
  EXPECT_EQ(stdc_bit_floor(static_cast<unsigned short>(0U)),
            static_cast<unsigned short>(0x5BU));
  EXPECT_EQ(stdc_bit_floor(0U), 0x5CU);
  EXPECT_EQ(stdc_bit_floor(0UL), 0x5DUL);
  EXPECT_EQ(stdc_bit_floor(0ULL), 0x5EULL);
}

TEST(LlvmLibcStdbitTest, TypeGenericMacroBitCeil) {
  EXPECT_EQ(stdc_bit_ceil(static_cast<unsigned char>(0U)),
            static_cast<unsigned char>(0x6AU));
  EXPECT_EQ(stdc_bit_ceil(static_cast<unsigned short>(0U)),
            static_cast<unsigned short>(0x6BU));
  EXPECT_EQ(stdc_bit_ceil(0U), 0x6CU);
  EXPECT_EQ(stdc_bit_ceil(0UL), 0x6DUL);
  EXPECT_EQ(stdc_bit_ceil(0ULL), 0x6EULL);
}

TEST(LlvmLibcStdbitTest, VersionMacro) {
  // 7.18.1p2 an integer constant expression with a value equivalent to 202311L.
  EXPECT_EQ(__STDC_VERSION_STDBIT_H__, 202311L);
}

TEST(LlvmLibcStdbitTest, EndianMacros) {
  // 7.18.2p3 The values of the integer constant expressions for
  // __STDC_ENDIAN_LITTLE__ and __STDC_ENDIAN_BIG__ are not equal.
  EXPECT_NE(__STDC_ENDIAN_LITTLE__, __STDC_ENDIAN_BIG__);
  // The standard does allow for __STDC_ENDIAN_NATIVE__ to be an integer
  // constant expression with an implementation defined value for non-big or
  // little endianness environments.  I assert such machines are no longer
  // relevant.
  EXPECT_TRUE(__STDC_ENDIAN_NATIVE__ == __STDC_ENDIAN_LITTLE__ ||
              __STDC_ENDIAN_NATIVE__ == __STDC_ENDIAN_BIG__);
}
