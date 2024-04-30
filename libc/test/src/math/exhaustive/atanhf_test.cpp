//===-- Exhaustive test for atanhf ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "exhaustive_test.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/atanhf.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

using FPBits = LIBC_NAMESPACE::fputil::FPBits<float>;

using LlvmLibcAtanhfExhaustiveTest =
    LlvmLibcUnaryOpExhaustiveMathTest<float, mpfr::Operation::Atanh,
                                      LIBC_NAMESPACE::atanhf>;

// Range: [0, 1.0];
static const uint32_t POS_START = 0x0000'0000U;
static const uint32_t POS_STOP = FPBits(1.0f).uintval();

TEST_F(LlvmLibcAtanhfExhaustiveTest, PostiveRange) {
  test_full_range_all_roundings(POS_START, POS_STOP);
}

// Range: [-1.0, 0];
static const uint32_t NEG_START = 0x8000'0000U;
static const uint32_t NEG_STOP = FPBits(-1.0f).uintval();

TEST_F(LlvmLibcAtanhfExhaustiveTest, NegativeRange) {
  test_full_range_all_roundings(NEG_START, NEG_STOP);
}
