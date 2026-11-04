//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Smoke test for the coshbf16 function.
///
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/bfloat16.h"
#include "src/math/coshbf16.h"
#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

class LlvmLibcCoshbf16Test : public LIBC_NAMESPACE::testing::FEnvSafeTest {
  DECLARE_SPECIAL_CONSTANTS(bfloat16)

public:
  void test_special_numbers() {
    EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::coshbf16(aNaN));
    // TODO: Add the remaining special values (sNaN, +-0, +-inf, overflow)
    // once coshbf16 is implemented.
  }
};

TEST_F(LlvmLibcCoshbf16Test, SpecialNumbers) { test_special_numbers(); }
