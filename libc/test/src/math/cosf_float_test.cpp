//===-- Unittests for cosf float-only -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/math/sincosf_float_eval.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

#include "hdr/stdint_proxy.h"

using LlvmLibcCosfFloatTest = LIBC_NAMESPACE::testing::FPTest<float>;

float cosf_fast(float x) {
  return LIBC_NAMESPACE::math::sincosf_float_eval::sincosf_eval<
      /*IS_SIN*/ false>(x);
}

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

TEST_F(LlvmLibcCosfFloatTest, InFloatRange) {
  constexpr uint32_t COUNT = 100'000;
  constexpr uint32_t STEP = UINT32_MAX / COUNT;
  for (uint32_t i = 0, v = 0; i <= COUNT; ++i, v += STEP) {
    float x = FPBits(v).get_val();
    if (FPBits(v).is_nan() || FPBits(v).is_inf())
      continue;
    ASSERT_MPFR_MATCH(mpfr::Operation::Cos, x, cosf_fast(x), 3.5);
  }
}
