//===-- Unittests for cargf -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/complex/cargf.h"

#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPCWrapper/MPCUtils.h"

using LlvmLibcCargTest = LIBC_NAMESPACE::testing::FPTest<float>;

namespace mpc = LIBC_NAMESPACE::testing::mpc;

TEST_F(LlvmLibcCargTest, RandomFloats) {
  _Complex float test1 = 5.0 + 10.0i;
  EXPECT_MPC_MATCH_ALL_ROUNDING(mpc::Operation::Carg, test1,
                           LIBC_NAMESPACE::cargf(test1), 0.5);
}
