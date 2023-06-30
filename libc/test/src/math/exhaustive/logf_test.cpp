//===-- Exhaustive test for logf ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "exhaustive_test.h"
#include "src/math/logf.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

namespace mpfr = __llvm_libc::testing::mpfr;

using LlvmLibcLogfExhaustiveTest =
    LlvmLibcUnaryOpExhaustiveMathTest<float, mpfr::Operation::Log,
                                      __llvm_libc::logf>;

// Range: [0, Inf];
static constexpr uint32_t POS_START = 0x0000'0000U;
static constexpr uint32_t POS_STOP = 0x7f80'0000U;

TEST_F(LlvmLibcLogfExhaustiveTest, PostiveRange) {
  test_full_range_all_roundings(POS_START, POS_STOP);
}
