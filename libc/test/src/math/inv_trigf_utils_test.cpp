//===-- Unittests for supfuncf --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "in_float_range_test_helper.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/generic/inv_trigf_utils.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include "utils/UnitTest/FPMatcher.h"
#include "utils/UnitTest/Test.h"
#include <math.h>

namespace mpfr = __llvm_libc::testing::mpfr;

DECLARE_SPECIAL_CONSTANTS(float)

constexpr int def_count = 100003;
constexpr float def_prec = 0.500001f;

auto f_normal = [](float x) -> bool { return !(isnan(x) || isinf(x)); };

TEST(LlvmLibcAtanfPosTest, InFloatRange) {
  CHECK_DATA(0.0f, inf, mpfr::Operation::Atan, __llvm_libc::atan_eval, f_normal,
             def_count, def_prec);
}

TEST(LlvmLibcAtanfNegTest, InFloatRange) {
  CHECK_DATA(-0.0f, neg_inf, mpfr::Operation::Atan, __llvm_libc::atan_eval,
             f_normal, def_count, def_prec);
}
