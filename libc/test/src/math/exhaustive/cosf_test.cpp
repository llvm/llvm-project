//===-- Exhaustive test for cosf ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/math/cosf.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include <math.h>

using FPBits = __llvm_libc::fputil::FPBits<float>;

namespace mpfr = __llvm_libc::testing::mpfr;

TEST(LlvmLibccosffExhaustiveTest, AllValues) {
  uint32_t bits = 0;
  do {
    FPBits xbits(bits);
    float x = float(xbits);
    ASSERT_MPFR_MATCH(mpfr::Operation::Cos, x, __llvm_libc::cosf(x), 1.0);
  } while (bits++ < 0xffff'ffffU);
}
