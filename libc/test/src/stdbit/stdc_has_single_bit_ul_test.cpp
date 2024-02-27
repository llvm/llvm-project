//===-- Unittests for stdc_has_single_bit_ul ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/limits.h"
#include "src/stdbit/stdc_has_single_bit_ul.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcStdcCountOnesUlTest, Zero) {
  EXPECT_EQ(LIBC_NAMESPACE::stdc_has_single_bit_ul(0U), false);
}

TEST(LlvmLibcStdcCountOnesUlTest, OneHot) {
  for (unsigned i = 0U; i != ULONG_WIDTH; ++i)
    EXPECT_EQ(LIBC_NAMESPACE::stdc_has_single_bit_ul(1UL << i), true);
}
