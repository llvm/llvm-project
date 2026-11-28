//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for exceptional value utilities (set_overflow / set_underflow).
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "hdr/math_macros.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/bfloat16.h"
#include "src/__support/FPUtil/except_value_utils.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/properties/types.h"
#include "src/__support/sign.h"
#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/RoundingModeUtils.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::Sign;
using LIBC_NAMESPACE::fputil::BFloat16;
using LIBC_NAMESPACE::fputil::FPBits;
using LIBC_NAMESPACE::fputil::raise_overflow_except_if_required;
using LIBC_NAMESPACE::fputil::raise_underflow_except_if_required;
using LIBC_NAMESPACE::fputil::set_overflow;
using LIBC_NAMESPACE::fputil::set_underflow;
using LIBC_NAMESPACE::fputil::testing::ForceRoundingMode;
using LIBC_NAMESPACE::fputil::testing::RoundingMode;

template <typename T>
class LlvmLibcExceptValueUtilsTest
    : public LIBC_NAMESPACE::testing::FEnvSafeTest {
  DECLARE_SPECIAL_CONSTANTS(T)

  template <typename Func>
  void check_call(RoundingMode mode, T expected, int expected_except,
                  Func &&func) {
    ForceRoundingMode r(mode);
    if (!r.success)
      return;

    libc_errno = 0;
    LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
    T actual = func();
    EXPECT_FP_EQ(expected, actual);
#ifndef LIBC_MATH_HAS_NO_EXCEPT
    if (math_errhandling & MATH_ERREXCEPT) {
      EXPECT_EQ(LIBC_NAMESPACE::fputil::test_except(FE_ALL_EXCEPT),
                expected_except);
    }
#else
    (void)expected_except;
#endif // LIBC_MATH_HAS_NO_EXCEPT
    EXPECT_MATH_ERRNO(ERANGE);
  }

  void check_all_roundings(T rn, T ru, T rd, T rz, int expected_except,
                           T (*fn_const)(), Sign runtime_sign,
                           bool is_overflow) {
    auto call_runtime = [&]() -> T {
      volatile bool neg = runtime_sign.is_neg();
      Sign s = neg ? Sign::NEG : Sign::POS;
      return is_overflow ? set_overflow<T>(s) : set_underflow<T>(s);
    };

    check_call(RoundingMode::Nearest, rn, expected_except, fn_const);
    check_call(RoundingMode::Nearest, rn, expected_except, call_runtime);
#ifndef LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
    check_call(RoundingMode::Upward, ru, expected_except, fn_const);
    check_call(RoundingMode::Upward, ru, expected_except, call_runtime);
    check_call(RoundingMode::Downward, rd, expected_except, fn_const);
    check_call(RoundingMode::Downward, rd, expected_except, call_runtime);
    check_call(RoundingMode::TowardZero, rz, expected_except, fn_const);
    check_call(RoundingMode::TowardZero, rz, expected_except, call_runtime);
#else
    (void)ru;
    (void)rd;
    (void)rz;
#endif
  }

public:
  void test_overflow() {
    constexpr int OVERFLOW_EXCEPTS = FE_OVERFLOW | FE_INEXACT;
    check_all_roundings(
        inf, inf, max_normal, max_normal, OVERFLOW_EXCEPTS,
        +[]() -> T { return set_overflow<T>(); }, Sign::POS,
        /*is_overflow=*/true);
    check_all_roundings(
        inf, inf, max_normal, max_normal, OVERFLOW_EXCEPTS,
        +[]() -> T { return set_overflow<T>(Sign::POS); }, Sign::POS,
        /*is_overflow=*/true);
    check_all_roundings(
        neg_inf, neg_max_normal, neg_inf, neg_max_normal, OVERFLOW_EXCEPTS,
        +[]() -> T { return set_overflow<T>(Sign::NEG); }, Sign::NEG,
        /*is_overflow=*/true);
  }

  void test_underflow() {
    constexpr int UNDERFLOW_EXCEPTS = FE_UNDERFLOW | FE_INEXACT;
    check_all_roundings(
        zero, min_denormal, zero, zero, UNDERFLOW_EXCEPTS,
        +[]() -> T { return set_underflow<T>(); }, Sign::POS,
        /*is_overflow=*/false);
    check_all_roundings(
        zero, min_denormal, zero, zero, UNDERFLOW_EXCEPTS,
        +[]() -> T { return set_underflow<T>(Sign::POS); }, Sign::POS,
        /*is_overflow=*/false);
    check_all_roundings(
        neg_zero, neg_zero, neg_min_denormal, neg_zero, UNDERFLOW_EXCEPTS,
        +[]() -> T { return set_underflow<T>(Sign::NEG); }, Sign::NEG,
        /*is_overflow=*/false);
  }

  void test_raise_except() {
    LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
    raise_overflow_except_if_required<T>();
#ifndef LIBC_MATH_HAS_NO_EXCEPT
    if (math_errhandling & MATH_ERREXCEPT) {
      EXPECT_EQ(LIBC_NAMESPACE::fputil::test_except(FE_ALL_EXCEPT),
                FE_OVERFLOW | FE_INEXACT);
    } else {
      EXPECT_EQ(LIBC_NAMESPACE::fputil::test_except(FE_ALL_EXCEPT), 0);
    }
#endif

    LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
    raise_underflow_except_if_required<T>();
#ifndef LIBC_MATH_HAS_NO_EXCEPT
    if (math_errhandling & MATH_ERREXCEPT) {
      EXPECT_EQ(LIBC_NAMESPACE::fputil::test_except(FE_ALL_EXCEPT),
                FE_UNDERFLOW | FE_INEXACT);
    } else {
      EXPECT_EQ(LIBC_NAMESPACE::fputil::test_except(FE_ALL_EXCEPT), 0);
    }
#endif
  }
};

using ExceptValueTypes =
    LIBC_NAMESPACE::testing::Types<float, double, long double,
#ifdef LIBC_TYPES_HAS_FLOAT16
                                   float16,
#endif // LIBC_TYPES_HAS_FLOAT16
#ifdef LIBC_TYPES_HAS_NATIVE_FLOAT128
                                   float128,
#endif // LIBC_TYPES_HAS_NATIVE_FLOAT128
                                   BFloat16>;

TYPED_TEST_SUITE(LlvmLibcExceptValueUtilsTest, ExceptValueTypes);

TYPED_TEST(LlvmLibcExceptValueUtilsTest, SetOverflow) { this->test_overflow(); }

TYPED_TEST(LlvmLibcExceptValueUtilsTest, SetUnderflow) {
  this->test_underflow();
}

TYPED_TEST(LlvmLibcExceptValueUtilsTest, RaiseExceptIfRequired) {
  this->test_raise_except();
}

#if LIBC_HAS_CONSTANT_EVALUATION
TEST(LlvmLibcExceptValueUtilsConstexprTest, ConstexprEval) {
  static_assert(FPBits<float>(set_overflow<float>()).uintval() ==
                FPBits<float>::inf(Sign::POS).uintval());
  static_assert(FPBits<float>(set_overflow<float>(Sign::NEG)).uintval() ==
                FPBits<float>::inf(Sign::NEG).uintval());
  static_assert(FPBits<float>(set_underflow<float>()).uintval() ==
                FPBits<float>::zero(Sign::POS).uintval());
  static_assert(FPBits<float>(set_underflow<float>(Sign::NEG)).uintval() ==
                FPBits<float>::zero(Sign::NEG).uintval());

  static_assert(FPBits<double>(set_overflow<double>()).uintval() ==
                FPBits<double>::inf(Sign::POS).uintval());
  static_assert(FPBits<double>(set_overflow<double>(Sign::NEG)).uintval() ==
                FPBits<double>::inf(Sign::NEG).uintval());
  static_assert(FPBits<double>(set_underflow<double>()).uintval() ==
                FPBits<double>::zero(Sign::POS).uintval());
  static_assert(FPBits<double>(set_underflow<double>(Sign::NEG)).uintval() ==
                FPBits<double>::zero(Sign::NEG).uintval());
}
#endif // LIBC_HAS_CONSTANT_EVALUATION
