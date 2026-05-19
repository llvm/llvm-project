//===-- Exhaustive test for SIMD expf -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "exhaustive_test.h"
#include "src/__support/CPP/simd.h"
#include "src/math/expf.h"
#include "src/mathvec/expf.h"

using LlvmLibcExpfExhaustiveTest =
    LlvmLibcUnaryOpExhaustiveMathvecTest<float, LIBC_NAMESPACE::expf,
                                         LIBC_NAMESPACE::expf>;

// Range: [0, Inf];
static constexpr uint32_t POS_START = 0x0000'0000U;
static constexpr uint32_t POS_STOP = 0x7fff'ffffU;

TEST_F(LlvmLibcExpfExhaustiveTest, PositiveRange) {
  test_full_range_RN(POS_START, POS_STOP);
}

// Range: [-Inf, 0];
static constexpr uint32_t NEG_START = 0xb000'0000U;
static constexpr uint32_t NEG_STOP = 0xffff'ffffU;

TEST_F(LlvmLibcExpfExhaustiveTest, NegativeRange) {
  test_full_range_RN(NEG_START, NEG_STOP);
}
