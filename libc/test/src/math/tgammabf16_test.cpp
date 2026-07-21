//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Exhaustive tests for the tgammabf16 function.
///
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/bfloat16.h"
#include "src/math/tgammabf16.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

using LlvmLibcTgammaBf16Test = LIBC_NAMESPACE::testing::FPTest<bfloat16>;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

static bool is_negative_integer(bfloat16 x) {
  float xf = static_cast<float>(x);
  if (xf > -1.0f)
    return false;
  if (xf <= -128.0f)
    return true;
  int n = static_cast<int>(xf);
  return xf == static_cast<float>(n);
}

TEST_F(LlvmLibcTgammaBf16Test, Exhaustive) {
  for (uint32_t v = 0x0000; v < 0x10000; ++v) {
    bfloat16 x =
        LIBC_NAMESPACE::fputil::FPBits<bfloat16>(static_cast<uint16_t>(v))
            .get_val();
    LIBC_NAMESPACE::fputil::FPBits<bfloat16> bits(x);

    if (bits.is_nan() || bits.is_zero() ||
        (bits.is_neg() && is_negative_integer(x)))
      continue;
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Tgamma, x,
                                   LIBC_NAMESPACE::tgammabf16(x), 1.0);
  }
}
