//===-- Unittests mod_2pi, mod_pi_over_4 and mod_pi_over_2 ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/generic/dp_trig.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

#include <math.h>

namespace mpfr = __llvm_libc::testing::mpfr;
using FPBits = __llvm_libc::fputil::FPBits<double>;
using UIntType = FPBits::UIntType;

TEST(LlvmLibcMod2PITest, Range) {
  constexpr UIntType count = 1000000000;
  constexpr UIntType step = UIntType(-1) / count;
  for (UIntType i = 0, v = 0; i <= count; ++i, v += step) {
    double x = double(FPBits(v));
    if (isnan(x) || isinf(x) || x <= 0.0)
      continue;

    ASSERT_MPFR_MATCH(mpfr::Operation::Mod2PI, x, __llvm_libc::mod_2pi(x), 0);
  }
}

TEST(LlvmLibcModPIOver2Test, Range) {
  constexpr UIntType count = 1000000000;
  constexpr UIntType step = UIntType(-1) / count;
  for (UIntType i = 0, v = 0; i <= count; ++i, v += step) {
    double x = double(FPBits(v));
    if (isnan(x) || isinf(x) || x <= 0.0)
      continue;

    ASSERT_MPFR_MATCH(mpfr::Operation::ModPIOver2, x,
                      __llvm_libc::mod_pi_over_2(x), 0);
  }
}

TEST(LlvmLibcModPIOver4Test, Range) {
  constexpr UIntType count = 1000000000;
  constexpr UIntType step = UIntType(-1) / count;
  for (UIntType i = 0, v = 0; i <= count; ++i, v += step) {
    double x = double(FPBits(v));
    if (isnan(x) || isinf(x) || x <= 0.0)
      continue;

    ASSERT_MPFR_MATCH(mpfr::Operation::ModPIOver4, x,
                      __llvm_libc::mod_pi_over_4(x), 0);
  }
}
