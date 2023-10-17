//===-- Unittests for supfuncf --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "in_float_range_test_helper.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/generic/explogxf.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include <math.h>

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

DECLARE_SPECIAL_CONSTANTS(float)

constexpr int def_count = 100003;
constexpr float def_prec = 0.500001f;

auto f_normal = [](float x) -> bool {
  return !(isnan(x) || isinf(x) || fabs(x) < 2E-38);
};

TEST(LlvmLibcExpxfTest, InFloatRange) {
  auto fx = [](float x) -> float {
    auto result = LIBC_NAMESPACE::exp_b_range_reduc<LIBC_NAMESPACE::ExpBase>(x);
    double r = LIBC_NAMESPACE::ExpBase::powb_lo(result.lo);
    return static_cast<float>(result.mh * r);
  };
  auto f_check = [](float x) -> bool {
    return !(
        (isnan(x) || isinf(x) || x < -70 || x > 70 || fabsf(x) < 0x1.0p-10));
  };
  CHECK_DATA(0.0f, neg_inf, mpfr::Operation::Exp, fx, f_check, def_count,
             def_prec);
}

TEST(LlvmLibcLog2xfTest, InFloatRange) {
  CHECK_DATA(0.0f, inf, mpfr::Operation::Log2, LIBC_NAMESPACE::log2_eval,
             f_normal, def_count, def_prec);
}

TEST(LlvmLibcLogxfTest, InFloatRange) {
  CHECK_DATA(0.0f, inf, mpfr::Operation::Log, LIBC_NAMESPACE::log_eval,
             f_normal, def_count, def_prec);
}
