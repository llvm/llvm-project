//===-- Unittests for atanf -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/math/atanf.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include <math.h>

#include <errno.h>
#include <stdint.h>

#include <initializer_list>

using FPBits = __llvm_libc::fputil::FPBits<float>;

namespace mpfr = __llvm_libc::testing::mpfr;

DECLARE_SPECIAL_CONSTANTS(float)

TEST(LlvmLibcAtanfTest, SpecialNumbers) {
  errno = 0;
  __llvm_libc::fputil::clear_except(FE_ALL_EXCEPT);
  EXPECT_FP_EQ(aNaN, __llvm_libc::atanf(aNaN));
  EXPECT_FP_EXCEPTION(0);
  EXPECT_MATH_ERRNO(0);

  __llvm_libc::fputil::clear_except(FE_ALL_EXCEPT);
  EXPECT_FP_EQ(0.0f, __llvm_libc::atanf(0.0f));
  EXPECT_FP_EXCEPTION(0);
  EXPECT_MATH_ERRNO(0);

  __llvm_libc::fputil::clear_except(FE_ALL_EXCEPT);
  EXPECT_FP_EQ(-0.0f, __llvm_libc::atanf(-0.0f));
  EXPECT_FP_EXCEPTION(0);
  EXPECT_MATH_ERRNO(0);
}

TEST(LlvmLibcAtanfTest, InFloatRange) {
  constexpr uint32_t COUNT = 1000000;
  const uint32_t STEP = FPBits(inf).uintval() / COUNT;
  for (uint32_t i = 0, v = 0; i <= COUNT; ++i, v += STEP) {
    float x = float(FPBits(v));
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Atan, x,
                                   __llvm_libc::atanf(x), 0.5);
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Atan, -x,
                                   __llvm_libc::atanf(-x), 0.5);
  }
}

// For small values, tanh(x) is x.
TEST(LlvmLibcAtanfTest, SpecialValues) {
  for (uint32_t v : {0x3d8d6b23U, 0x3feefcfbU, 0xbd8d6b23U, 0xbfeefcfbU,
                     0x7F800000U, 0xFF800000U}) {
    float x = float(FPBits(v));
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Atan, x,
                                   __llvm_libc::atanf(x), 0.5);
  }
}
