//===-- Exhaustive test for acoshf ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "exhaustive_test.h"
#include "src/math/acoshf.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

namespace mpfr = __llvm_libc::testing::mpfr;

using LlvmLibcAcoshfExhaustiveTest =
    LlvmLibcUnaryOpExhaustiveMathTest<float, mpfr::Operation::Acosh,
                                      __llvm_libc::acoshf>;

// Range: [1, Inf];
static constexpr uint32_t POS_START = 0x3f80'0000U;
static constexpr uint32_t POS_STOP = 0x7f80'0000U;

TEST_F(LlvmLibcAcoshfExhaustiveTest, PostiveRangeRound) {
  test_full_range_all_roundings(POS_START, POS_STOP);
}
