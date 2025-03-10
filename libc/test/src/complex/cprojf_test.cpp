//===-- Unittests for cprojf ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CprojTest.h"

#include "src/complex/cprojf.h"

#include "utils/MPCWrapper/MPCUtils.h"

using LlvmLibcCprojTestMPC = LIBC_NAMESPACE::testing::FPTest<float>;

namespace mpc = LIBC_NAMESPACE::testing::mpc;

TEST_F(LlvmLibcCprojTestMPC, MPCRND) {
  _Complex float test = 5.0 + 10.0i;
  EXPECT_MPC_MATCH_ALL_ROUNDING(mpc::Operation::Cproj, test,
                                LIBC_NAMESPACE::cprojf(test), 0.5);
}

LIST_CPROJ_TESTS(_Complex float, float, LIBC_NAMESPACE::cprojf)
