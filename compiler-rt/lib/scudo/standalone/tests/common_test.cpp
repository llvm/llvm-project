//===-- common_test.cpp -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "internal_defs.h"
#include "tests/scudo_unit_test.h"

#include "common.h"
#include "mem_map.h"

#include <errno.h>
#include <string.h>
#include <sys/mman.h>

namespace scudo {

TEST(ScudoCommonTest, IsPowerOfTwo) {
  EXPECT_FALSE(isPowerOfTwo(0));
  EXPECT_TRUE(isPowerOfTwo(1));
  EXPECT_TRUE(isPowerOfTwo(2));
  EXPECT_TRUE(isPowerOfTwo(4));
  EXPECT_FALSE(isPowerOfTwo(3));
}

TEST(ScudoCommonTest, ComputePercentage) {
  uptr Integral, Fractional;
  computePercentage(50, 100, &Integral, &Fractional);
  EXPECT_EQ(Integral, 50U);
  EXPECT_EQ(Fractional, 0U);

  computePercentage(1, 3, &Integral, &Fractional);
  EXPECT_EQ(Integral, 33U);
  EXPECT_EQ(Fractional, 33U);

  computePercentage(2, 3, &Integral, &Fractional);
  EXPECT_EQ(Integral, 66U);
  EXPECT_EQ(Fractional, 67U);

  computePercentage(0, 0, &Integral, &Fractional);
  EXPECT_EQ(Integral, 100U);
  EXPECT_EQ(Fractional, 0U);
}

TEST(ScudoCommonTest, IsAlignedSlow) {
  EXPECT_TRUE(isAlignedSlow(64, 16));
  EXPECT_FALSE(isAlignedSlow(65, 16));
}

} // namespace scudo
