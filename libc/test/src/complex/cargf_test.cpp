//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains unittest for cargf.
///
//===----------------------------------------------------------------------===//

#include "CArgTest.h"

#include "src/complex/cargf.h"
#include "utils/MPCWrapper/MPCUtils.h"

using LlvmLibcCargfMPCTest = LIBC_NAMESPACE::testing::FPTest<float>;

namespace mpc = LIBC_NAMESPACE::testing::mpc;

TEST_F(LlvmLibcCargfMPCTest, BasicForRounding) {
  _Complex float test_values[] = {
      1.0f + 1.0fi,  -1.0f + 1.0fi,  1.0f - 1.0fi,      -1.0f - 1.0fi,
      3.0f + 4.0fi,  -3.0f + 4.0fi,  3.0f - 4.0fi,      -3.0f - 4.0fi,
      0.5f + 0.5fi,  -0.5f + 0.5fi,  0.5f - 0.5fi,      -0.5f - 0.5fi,
      1.0f + 0.0fi,  -1.0f + 0.0fi,  0.0f + 1.0fi,      0.0f - 1.0fi,
      5.0f + 12.0fi, 100.0f + 1.0fi, 0.001f + 1000.0fi,
  };
  for (_Complex float val : test_values) {
    EXPECT_MPC_MATCH_ALL_ROUNDING(mpc::Operation::Carg, val,
                                  LIBC_NAMESPACE::cargf(val), 0.5);
  }
}

LIST_CARG_TESTS(_Complex float, float, LIBC_NAMESPACE::cargf)
