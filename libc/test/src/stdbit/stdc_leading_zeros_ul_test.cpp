//===-- Unittests for stdc_leading_zeros_ul -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdbit/stdc_leading_zeros_ul.h"
#include "src/__support/CPP/limits.h"
#include "test/UnitTest/Test.h"
#include <stddef.h>

#define LZ(x) LIBC_NAMESPACE::stdc_leading_zeros_ul((x))

TEST(LlvmLibcStdcLeadingZerosUlTest, Zero) {
  EXPECT_EQ(LZ(0UL), static_cast<unsigned long>(ULONG_WIDTH));
}

TEST(LlvmLibcStdcLeadingZerosUlTest, OneHot) {
  for (unsigned i = 0U; i != ULONG_WIDTH; ++i)
    EXPECT_EQ(LZ(1UL << i), static_cast<unsigned long>(ULONG_WIDTH - i - 1));
}
