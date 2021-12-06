//===-- Exhaustive test for logf ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/math/logf.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include "utils/UnitTest/FPMatcher.h"
#include "utils/UnitTest/Test.h"

using FPBits = __llvm_libc::fputil::FPBits<float>;

namespace mpfr = __llvm_libc::testing::mpfr;

TEST(LlvmLibcLogfExhaustiveTest, AllValues) {
  uint32_t bits = 0U;
  do {
    FPBits xbits(bits);
    float x = float(xbits);
    EXPECT_MPFR_MATCH(mpfr::Operation::Log, x, __llvm_libc::logf(x), 0.5);
  } while (bits++ < 0x7f7f'ffffU);
}
