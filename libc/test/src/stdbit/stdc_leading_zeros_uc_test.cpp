//===-- Unittests for stdc_leading_zeros_uc -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/limits.h"
#include "src/stdbit/stdc_leading_zeros_uc.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcStdcLeadingZerosUcTest, Zero) {
  EXPECT_EQ(LIBC_NAMESPACE::stdc_leading_zeros_uc(0U),
            static_cast<unsigned char>(UCHAR_WIDTH));
}

TEST(LlvmLibcStdcLeadingZerosUcTest, OneHot) {
  for (unsigned i = 0U; i != UCHAR_WIDTH; ++i)
    EXPECT_EQ(LIBC_NAMESPACE::stdc_leading_zeros_uc(1U << i),
              static_cast<unsigned char>(UCHAR_WIDTH - i - 1));
}
