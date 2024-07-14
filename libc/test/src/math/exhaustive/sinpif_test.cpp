//===-- Exhaustive test for sinpif ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "exhaustive_test.h"
#include "mpfr.h"
#include "src/math/sinpif.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include <sys/types.h>

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

using LlvmLibcSinpifExhaustiveTest =
    LlvmLibcUnaryOpExhaustiveMathTest<float, mpfr::Operation::Sinpi,
                                      LIBC_NAMESPACE::sinpif>;

static constexpr uint32_t POS_START = 0x0000'0000U;
static constexpr uint32_t POS_STOP = 0x7f80'0000U;

// Range: [0, Inf]
TEST_F(LlvmLibcSinpifExhaustiveTest, PostiveRange) {
  test_full_range_all_roundings(POS_START, POS_STOP);
}

// Range: [-Inf, 0]
static constexpr uint32_t NEG_START = 0xb000'0000U;
static constexpr uint32_t NEG_STOP = 0xff80'0000U;

TEST_F(LlvmLibcSinpifExhaustiveTest, NegativeRange) {
  test_full_range_all_roundings(NEG_START, NEG_STOP);
}
