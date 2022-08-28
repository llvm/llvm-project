//===-- Unittests for explogxf --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "in_float_range_test_helper.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/generic/explogxf.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include "utils/UnitTest/FPMatcher.h"
#include "utils/UnitTest/Test.h"
#include <math.h>

namespace mpfr = __llvm_libc::testing::mpfr;

DECLARE_SPECIAL_CONSTANTS(float)

constexpr int def_count = 100003;
constexpr float def_prec = 0.500001f;

TEST(LlvmLibcExpxfTest, InFloatRange) {
  auto fx = [](float x) -> float {
    auto result = __llvm_libc::exp_eval<-1>(x);
    return static_cast<float>(2 * result.mult_exp * result.r +
                              2 * result.mult_exp);
  };
  auto f_check = [](float x) -> bool {
    return !(
        (isnan(x) || isinf(x) || x < -70 || x > 70 || fabsf(x) < 0x1.0p-10));
  };
  CHECK_DATA(0.0f, neg_inf, mpfr::Operation::Exp, fx, f_check, def_count,
             def_prec);
}

TEST(LlvmLibcExp2xfTest, InFloatRange) {
  auto f_check = [](float x) -> bool {
    return !(
        (isnan(x) || isinf(x) || x < -130 || x > 130 || fabsf(x) < 0x1.0p-10));
  };
  CHECK_DATA(0.0f, neg_inf, mpfr::Operation::Exp2, __llvm_libc::exp2_eval,
             f_check, def_count, def_prec);
}

TEST(LlvmLibcLog2xfTest, InFloatRange) {
  CHECK_DATA(0.0f, inf, mpfr::Operation::Log2, __llvm_libc::log2_eval, isnormal,
             def_count, def_prec);
}

TEST(LlvmLibcLogxfTest, InFloatRange) {
  CHECK_DATA(0.0f, inf, mpfr::Operation::Log, __llvm_libc::log_eval, isnormal,
             def_count, def_prec);
}
