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
unsigned char stdc_leading_zeros_uc(unsigned char) noexcept { return 0xAA; }
unsigned short stdc_leading_zeros_us(unsigned short) noexcept { return 0xAB; }
unsigned stdc_leading_zeros_ui(unsigned) noexcept { return 0xAC; }
unsigned long stdc_leading_zeros_ul(unsigned long) noexcept { return 0xAD; }
unsigned long long stdc_leading_zeros_ull(unsigned long long) noexcept {
  return 0xAF;
}
unsigned char stdc_leading_ones_uc(unsigned char) noexcept { return 0xBA; }
unsigned short stdc_leading_ones_us(unsigned short) noexcept { return 0xBB; }
unsigned stdc_leading_ones_ui(unsigned) noexcept { return 0xBC; }
unsigned long stdc_leading_ones_ul(unsigned long) noexcept { return 0xBD; }
unsigned long long stdc_leading_ones_ull(unsigned long long) noexcept {
  return 0xBF;
}
}

#include "include/llvm-libc-macros/stdbit-macros.h"

TEST(LlvmLibcStdbitTest, TypeGenericMacroLeadingZeros) {
  EXPECT_EQ(stdc_leading_zeros(static_cast<unsigned char>(0U)),
            static_cast<unsigned char>(0xAA));
  EXPECT_EQ(stdc_leading_zeros(static_cast<unsigned short>(0U)),
            static_cast<unsigned short>(0xAB));
  EXPECT_EQ(stdc_leading_zeros(0U), static_cast<unsigned>(0xAC));
  EXPECT_EQ(stdc_leading_zeros(0UL), static_cast<unsigned long>(0xAD));
  EXPECT_EQ(stdc_leading_zeros(0ULL), static_cast<unsigned long long>(0xAF));
}

TEST(LlvmLibcStdbitTest, TypeGenericMacroLeadingOnes) {
  EXPECT_EQ(stdc_leading_ones(static_cast<unsigned char>(0U)),
            static_cast<unsigned char>(0xBA));
  EXPECT_EQ(stdc_leading_ones(static_cast<unsigned short>(0U)),
            static_cast<unsigned short>(0xBB));
  EXPECT_EQ(stdc_leading_ones(0U), static_cast<unsigned>(0xBC));
  EXPECT_EQ(stdc_leading_ones(0UL), static_cast<unsigned long>(0xBD));
  EXPECT_EQ(stdc_leading_ones(0ULL), static_cast<unsigned long long>(0xBF));
}
