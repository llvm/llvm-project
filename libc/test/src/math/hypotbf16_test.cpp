//===-- Exhaustive test for hypotbf16 -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "HypotTest.h"

#include "src/math/hypotbf16.h"
#include "src/__support/FPUtil/bfloat16.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

using LlvmLibcHypotBf16Test = LIBC_NAMESPACE::testing::FPTest<bfloat16>;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

// range: [+0, inf]
static constexpr uint16_t POS_START = 0x0000U;
static constexpr uint16_t POS_STOP = 0x7f80U;

// x+y = y+x
TEST_F(LlvmLibcHypotBf16Test, PositiveRange){
    for(uint16_t v1 = POS_START; v1<= POS_STOP; ++v1){
        for(uint16_t v2 = v1; v2<=POS_STOP; ++v2){
            bfloat16 x = FPBits(x).get_val();
            bfloat16 y = FPBits(y).get_val();
            mpfr::BinaryInput<bfloat16> input{x,y};
            EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Hypot, input,LIBC_NAMESPACE::hypotbf16(x, y),0.5);
        }
    }
}

TEST_F(LlvmLibcHypotBf16Test, SpecialNumbers){
    constexpr bfloat16 VAL[] = {zero,    neg_zero,   inf,
                                   neg_inf, min_normal, max_normal};
    for(size_t v1 = 0; v1< 6; ++v1){
        for(uint16_t v2 = v1 ; v2< 6; ++v2){

            bfloat16 x = VAL[v1];
            bfloat16 y = VAL[v2];
            mpfr::BinaryInput<bfloat16> input{x,y};
            EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Hypot, input,LIBC_NAMESPACE::hypotbf16(x, y),0.5);
        }
    }
}



