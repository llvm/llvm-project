//===-- Unittests for stdbit ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDSList-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "test/UnitTest/Test.h"

#include <stdbit.h>

/*
 * The intent of this test is validate that:
 * 1. We provide the definition of the various type generic macros of stdbit.h.
 * 2. It dispatches to the correct underlying function.
 * Because unit tests build without public packaging, the object files produced
 * do not contain non-namespaced symbols.
 */

unsigned char stdc_leading_zeros_uc(unsigned char) { return 0xAA; }
unsigned short stdc_leading_zeros_us(unsigned short) { return 0xAB; }
unsigned stdc_leading_zeros_ui(unsigned) { return 0xAC; }
unsigned long stdc_leading_zeros_ul(unsigned long) { return 0xAD; }
unsigned long long stdc_leading_zeros_ull(unsigned long long) { return 0xAF; }

TEST(LlvmLibcStdbitTest, TypeGenericMacro) {
  EXPECT_EQ(stdc_leading_zeros(static_cast<unsigned char>(0U)),
            static_cast<unsigned char>(0xAA));
  EXPECT_EQ(stdc_leading_zeros(static_cast<unsigned short>(0U)),
            static_cast<unsigned short>(0xAB));
  EXPECT_EQ(stdc_leading_zeros(0U), static_cast<unsigned>(0xAC));
  EXPECT_EQ(stdc_leading_zeros(0UL), static_cast<unsigned long>(0xAD));
  EXPECT_EQ(stdc_leading_zeros(0ULL), static_cast<unsigned long long>(0xAF));
}
