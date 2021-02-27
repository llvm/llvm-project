//===-- Exhaustive test for sqrtf -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "src/math/sqrtf.h"
#include "utils/FPUtil/FPBits.h"
#include "utils/FPUtil/TestHelpers.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include <math.h>

using FPBits = __llvm_libc::fputil::FPBits<float>;

namespace mpfr = __llvm_libc::testing::mpfr;

TEST(LlvmLibcSqrtfExhaustiveTest, AllValues) {
  uint32_t bits = 0;
  do {
    FPBits x(bits);
    ASSERT_MPFR_MATCH(mpfr::Operation::Sqrt, float(x), __llvm_libc::sqrtf(x),
                      0.5);
  } while (bits++ < 0xffff'ffffU);
}
