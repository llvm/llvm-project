//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains unittest for carg.
///
//===----------------------------------------------------------------------===//

#include "CArgTest.h"

#include "src/complex/carg.h"
#include "utils/MPCWrapper/MPCUtils.h"

using LlvmLibcCargMPCTest = LIBC_NAMESPACE::testing::FPTest<double>;

namespace mpc = LIBC_NAMESPACE::testing::mpc;

TEST_F(LlvmLibcCargMPCTest, BasicForRounding) {
  _Complex double test_values[] = {
      1.0 + 1.0i,  -1.0 + 1.0i, 1.0 - 1.0i,   -1.0 - 1.0i,     3.0 + 4.0i,
      -3.0 + 4.0i, 3.0 - 4.0i,  -3.0 - 4.0i,  0.5 + 0.5i,      -0.5 + 0.5i,
      0.5 - 0.5i,  -0.5 - 0.5i, 1.0 + 0.0i,   -1.0 + 0.0i,     0.0 + 1.0i,
      0.0 - 1.0i,  5.0 + 12.0i, 100.0 + 1.0i, 0.001 + 1000.0i,
  };
  for (_Complex double val : test_values) {
    EXPECT_MPC_MATCH_ALL_ROUNDING(mpc::Operation::Carg, val,
                                  LIBC_NAMESPACE::carg(val), 0.5);
  }
}

LIST_CARG_TESTS(_Complex double, double, LIBC_NAMESPACE::carg)
