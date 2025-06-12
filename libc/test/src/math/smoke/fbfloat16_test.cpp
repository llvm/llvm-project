//===-- Unittests for fbfloat16 function ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/bfloat16.h"

#include "src/math/fbfloat16.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcFBfloat16Test = LIBC_NAMESPACE::testing::FPTest<bfloat16>;

TEST_F(LlvmLibcFBfloat16Test, SpecialNumbers) {
  constexpr float SPECIAL_FLOATS[] = {
    0.0f, 1.0f, 2.0f, 4.5f, -1.0f, -0.5f, 3.140625f
  };

  constexpr uint16_t SPECIAL_BFLOAT16_BITS[] = {
    0, 0x3f80U, 0x4000U, 0x4090U, 0xbf80U, 0xbf00, 0x4049U
  };

  for (int i=0; i<7; i++) {
    bfloat16 x{SPECIAL_FLOATS[i]};
    ASSERT_EQ(SPECIAL_BFLOAT16_BITS[i], x.bits);
  }
}
