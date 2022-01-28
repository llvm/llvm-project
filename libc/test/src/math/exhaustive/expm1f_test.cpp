//===-- Exhaustive test for expm1f-----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/math/expm1f.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include <math.h>

using FPBits = __llvm_libc::fputil::FPBits<float>;

namespace mpfr = __llvm_libc::testing::mpfr;

TEST(LlvmLibcExpm1fExhaustiveTest, AllValues) {
  uint32_t bits = 0;
  do {
    FPBits x(bits);
    if (!x.is_inf_or_nan() && float(x) < 88.70f) {
      ASSERT_MPFR_MATCH(mpfr::Operation::Expm1, float(x),
                        __llvm_libc::expm1f(float(x)), 1.5);
    }
  } while (bits++ < 0xffff'ffffU);
}
