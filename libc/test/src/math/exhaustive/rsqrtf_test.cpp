//===-- Exhaustive test for rsqrtf ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "exhaustive_test.h"
#include "src/math/rsqrtf.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

using LlvmLibcRsqrtfTest = LIBC_NAMESPACE::testing::FPTest<float>;

using LlvmLibcRsqrtfExhaustiveTest =
    LlvmLibcUnaryOpExhaustiveMathTest<float, mpfr::Operation::Rsqrt,
                                      LIBC_NAMESPACE::rsqrtf>;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

// Range: [0, Inf]
static constexpr uint32_t POS_START = 0x0000'0000U;
static constexpr uint32_t POS_STOP = 0x7f80'0000U;

TEST_F(LlvmLibcRsqrtfExhaustiveTest, PositiveRange) {
  test_full_range_all_roundings(POS_START, POS_STOP);
}
