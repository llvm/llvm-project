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
extern "C" {
unsigned stdc_leading_zeros_uc(unsigned char) noexcept { return 0xAAU; }
unsigned stdc_leading_zeros_us(unsigned short) noexcept { return 0xABU; }
unsigned stdc_leading_zeros_ui(unsigned) noexcept { return 0xACU; }
unsigned stdc_leading_zeros_ul(unsigned long) noexcept { return 0xADU; }
unsigned stdc_leading_zeros_ull(unsigned long long) noexcept { return 0xAFU; }
unsigned stdc_leading_ones_uc(unsigned char) noexcept { return 0xBAU; }
unsigned stdc_leading_ones_us(unsigned short) noexcept { return 0xBBU; }
unsigned stdc_leading_ones_ui(unsigned) noexcept { return 0xBCU; }
unsigned stdc_leading_ones_ul(unsigned long) noexcept { return 0xBDU; }
unsigned stdc_leading_ones_ull(unsigned long long) noexcept { return 0xBFU; }
unsigned stdc_trailing_zeros_uc(unsigned char) noexcept { return 0xCAU; }
unsigned stdc_trailing_zeros_us(unsigned short) noexcept { return 0xCBU; }
unsigned stdc_trailing_zeros_ui(unsigned) noexcept { return 0xCCU; }
unsigned stdc_trailing_zeros_ul(unsigned long) noexcept { return 0xCDU; }
unsigned stdc_trailing_zeros_ull(unsigned long long) noexcept { return 0xCFU; }
unsigned stdc_trailing_ones_uc(unsigned char) noexcept { return 0xDAU; }
unsigned stdc_trailing_ones_us(unsigned short) noexcept { return 0xDBU; }
unsigned stdc_trailing_ones_ui(unsigned) noexcept { return 0xDCU; }
unsigned stdc_trailing_ones_ul(unsigned long) noexcept { return 0xDDU; }
unsigned stdc_trailing_ones_ull(unsigned long long) noexcept { return 0xDFU; }
unsigned stdc_first_leading_zero_uc(unsigned char) noexcept { return 0xEAU; }
unsigned stdc_first_leading_zero_us(unsigned short) noexcept { return 0xEBU; }
unsigned stdc_first_leading_zero_ui(unsigned) noexcept { return 0xECU; }
unsigned stdc_first_leading_zero_ul(unsigned long) noexcept { return 0xEDU; }
unsigned stdc_first_leading_zero_ull(unsigned long long) noexcept {
  return 0xEFU;
}
unsigned stdc_first_leading_one_uc(unsigned char) noexcept { return 0xFAU; }
unsigned stdc_first_leading_one_us(unsigned short) noexcept { return 0xFBU; }
unsigned stdc_first_leading_one_ui(unsigned) noexcept { return 0xFCU; }
unsigned stdc_first_leading_one_ul(unsigned long) noexcept { return 0xFDU; }
unsigned stdc_first_leading_one_ull(unsigned long long) noexcept {
  return 0xFFU;
}
unsigned stdc_first_trailing_zero_uc(unsigned char) noexcept { return 0x0AU; }
unsigned stdc_first_trailing_zero_us(unsigned short) noexcept { return 0x0BU; }
unsigned stdc_first_trailing_zero_ui(unsigned) noexcept { return 0x0CU; }
unsigned stdc_first_trailing_zero_ul(unsigned long) noexcept { return 0x0DU; }
unsigned stdc_first_trailing_zero_ull(unsigned long long) noexcept {
  return 0x0FU;
}
unsigned stdc_first_trailing_one_uc(unsigned char) noexcept { return 0x1AU; }
unsigned stdc_first_trailing_one_us(unsigned short) noexcept { return 0x1BU; }
unsigned stdc_first_trailing_one_ui(unsigned) noexcept { return 0x1CU; }
unsigned stdc_first_trailing_one_ul(unsigned long) noexcept { return 0x1DU; }
unsigned stdc_first_trailing_one_ull(unsigned long long) noexcept {
  return 0x1FU;
}
}

#include "include/llvm-libc-macros/stdbit-macros.h"

TEST(LlvmLibcStdbitTest, TypeGenericMacroLeadingZeros) {
  EXPECT_EQ(stdc_leading_zeros(static_cast<unsigned char>(0U)), 0xAAU);
  EXPECT_EQ(stdc_leading_zeros(static_cast<unsigned short>(0U)), 0xABU);
  EXPECT_EQ(stdc_leading_zeros(0U), 0xACU);
  EXPECT_EQ(stdc_leading_zeros(0UL), 0xADU);
  EXPECT_EQ(stdc_leading_zeros(0ULL), 0xAFU);
}

TEST(LlvmLibcStdbitTest, TypeGenericMacroLeadingOnes) {
  EXPECT_EQ(stdc_leading_ones(static_cast<unsigned char>(0U)), 0xBAU);
  EXPECT_EQ(stdc_leading_ones(static_cast<unsigned short>(0U)), 0xBBU);
  EXPECT_EQ(stdc_leading_ones(0U), 0xBCU);
  EXPECT_EQ(stdc_leading_ones(0UL), 0xBDU);
  EXPECT_EQ(stdc_leading_ones(0ULL), 0xBFU);
}

TEST(LlvmLibcStdbitTest, TypeGenericMacroTrailingZeros) {
  EXPECT_EQ(stdc_trailing_zeros(static_cast<unsigned char>(0U)), 0xCAU);
  EXPECT_EQ(stdc_trailing_zeros(static_cast<unsigned short>(0U)), 0xCBU);
  EXPECT_EQ(stdc_trailing_zeros(0U), 0xCCU);
  EXPECT_EQ(stdc_trailing_zeros(0UL), 0xCDU);
  EXPECT_EQ(stdc_trailing_zeros(0ULL), 0xCFU);
}

TEST(LlvmLibcStdbitTest, TypeGenericMacroTrailingOnes) {
  EXPECT_EQ(stdc_trailing_ones(static_cast<unsigned char>(0U)), 0xDAU);
  EXPECT_EQ(stdc_trailing_ones(static_cast<unsigned short>(0U)), 0xDBU);
  EXPECT_EQ(stdc_trailing_ones(0U), 0xDCU);
  EXPECT_EQ(stdc_trailing_ones(0UL), 0xDDU);
  EXPECT_EQ(stdc_trailing_ones(0ULL), 0xDFU);
}

TEST(LlvmLibcStdbitTest, TypeGenericMacroFirstLeadingZero) {
  EXPECT_EQ(stdc_first_leading_zero(static_cast<unsigned char>(0U)), 0xEAU);
  EXPECT_EQ(stdc_first_leading_zero(static_cast<unsigned short>(0U)), 0xEBU);
  EXPECT_EQ(stdc_first_leading_zero(0U), 0xECU);
  EXPECT_EQ(stdc_first_leading_zero(0UL), 0xEDU);
  EXPECT_EQ(stdc_first_leading_zero(0ULL), 0xEFU);
}

TEST(LlvmLibcStdbitTest, TypeGenericMacroFirstLeadingOne) {
  EXPECT_EQ(stdc_first_leading_one(static_cast<unsigned char>(0U)), 0xFAU);
  EXPECT_EQ(stdc_first_leading_one(static_cast<unsigned short>(0U)), 0xFBU);
  EXPECT_EQ(stdc_first_leading_one(0U), 0xFCU);
  EXPECT_EQ(stdc_first_leading_one(0UL), 0xFDU);
  EXPECT_EQ(stdc_first_leading_one(0ULL), 0xFFU);
}

TEST(LlvmLibcStdbitTest, TypeGenericMacroFirstTrailingZero) {
  EXPECT_EQ(stdc_first_trailing_zero(static_cast<unsigned char>(0U)), 0x0AU);
  EXPECT_EQ(stdc_first_trailing_zero(static_cast<unsigned short>(0U)), 0x0BU);
  EXPECT_EQ(stdc_first_trailing_zero(0U), 0x0CU);
  EXPECT_EQ(stdc_first_trailing_zero(0UL), 0x0DU);
  EXPECT_EQ(stdc_first_trailing_zero(0ULL), 0x0FU);
}

TEST(LlvmLibcStdbitTest, TypeGenericMacroFirstTrailingOne) {
  EXPECT_EQ(stdc_first_trailing_one(static_cast<unsigned char>(0U)), 0x1AU);
  EXPECT_EQ(stdc_first_trailing_one(static_cast<unsigned short>(0U)), 0x1BU);
  EXPECT_EQ(stdc_first_trailing_one(0U), 0x1CU);
  EXPECT_EQ(stdc_first_trailing_one(0UL), 0x1DU);
  EXPECT_EQ(stdc_first_trailing_one(0ULL), 0x1FU);
}
