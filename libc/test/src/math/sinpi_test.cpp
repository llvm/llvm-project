//===-- Exhaustive test for sinpif16 --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "src/math/sinpi.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

#include <iostream>

using LlvmLibcSinpiTest = LIBC_NAMESPACE::testing::FPTest<double>;
using namespace std;
namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

static constexpr uint64_t POS_START = 0x0000U;
//static constexpr uint64_t POS_STOP =  0x7c00U;

#include <iostream>
/*
double inputs[] = {
  0x1p0, 0x1p1, 0x1.8p1, 0x1p2, 0x1.4p2, 0x1.8p2, 
  0x1.cp2, 0x1p3, 0x1.2p3, 0x1.4p3, 0x1.6p3, 0x1.8p3, 
  0x1.ap3, 0x1.cp3, 0x1.ep3, 0x1p4, 0x1.1p4, 0x1.2p4, 
  0x1.3p4, 0x1.4p4, 0x1.5p4, 0x1.6p4, 0x1.7p4, 0x1.8p4, 
 0x1.9p4, 0x1.ap4
};
*/

TEST_F(LlvmLibcSinpiTest, PositiveRange) {
  for (uint64_t v = POS_START; v <= 3; ++v) {
    double x = FPBits(v).get_val();
    std::cout << "sin(" << x << " * pi) =" << "\n" <<  LIBC_NAMESPACE::sinpi(x) << std::endl;
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Sinpi, x,
				   LIBC_NAMESPACE::sinpi(x), 0.5);
    
    //EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Sinpi, -v,
    //LIBC_NAMESPACE::sinpi(-v), 0.5);
  }
}
