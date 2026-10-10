//===-- unittests/Runtime/Complex.cpp ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gmock/gmock.h"
#include "gtest/gtest-matchers.h"
#include <cstring>
#include <limits>

#ifdef __clang__
#pragma clang diagnostic ignored "-Wc99-extensions"
#endif

#include "flang/Common/Fortran-consts.h"
#include "flang/Runtime/cpp-type.h"
#include "flang/Runtime/entry-names.h"

#include <complex>
#include <cstdint>

#ifndef _MSC_VER
#include <complex.h>
typedef float _Complex float_Complex_t;
typedef double _Complex double_Complex_t;
#else
struct float_Complex_t {
  float re;
  float im;
};
struct double_Complex_t {
  double re;
  double im;
};
#endif

extern "C" float_Complex_t RTNAME(cpowi)(
    float_Complex_t base, std::int32_t exp);

extern "C" double_Complex_t RTNAME(zpowi)(
    double_Complex_t base, std::int32_t exp);

extern "C" float_Complex_t RTNAME(cpowk)(
    float_Complex_t base, std::int64_t exp);

extern "C" double_Complex_t RTNAME(zpowk)(
    double_Complex_t base, std::int64_t exp);

static std::complex<float> cpowi(std::complex<float> base, std::int32_t exp) {
  float_Complex_t cbase{*(float_Complex_t *)(&base)};
  float_Complex_t cres{RTNAME(cpowi)(cbase, exp)};
  return *(std::complex<float> *)(&cres);
}

static std::complex<double> zpowi(std::complex<double> base, std::int32_t exp) {
  double_Complex_t cbase{*(double_Complex_t *)(&base)};
  double_Complex_t cres{RTNAME(zpowi)(cbase, exp)};
  return *(std::complex<double> *)(&cres);
}

static std::complex<float> cpowk(std::complex<float> base, std::int64_t exp) {
  float_Complex_t cbase{*(float_Complex_t *)(&base)};
  float_Complex_t cres{RTNAME(cpowk)(cbase, exp)};
  return *(std::complex<float> *)(&cres);
}

static std::complex<double> zpowk(std::complex<double> base, std::int64_t exp) {
  double_Complex_t cbase{*(double_Complex_t *)(&base)};
  double_Complex_t cres{RTNAME(zpowk)(cbase, exp)};
  return *(std::complex<double> *)(&cres);
}

MATCHER_P(ExpectComplexFloatEq, c, "") {
  using namespace testing;
  return ExplainMatchResult(
      AllOf(Property(&std::complex<float>::real, FloatEq(c.real())),
          Property(&std::complex<float>::imag, FloatEq(c.imag()))),
      arg, result_listener);
}

MATCHER_P(ExpectComplexDoubleEq, c, "") {
  using namespace testing;
  return ExplainMatchResult(AllOf(Property(&std::complex<double>::real,
                                      DoubleNear(c.real(), 0.00000001)),
                                Property(&std::complex<double>::imag,
                                    DoubleNear(c.imag(), 0.00000001))),
      arg, result_listener);
}

#define EXPECT_COMPLEX_FLOAT_EQ(val1, val2) \
  EXPECT_THAT(val1, ExpectComplexFloatEq(val2))

#define EXPECT_COMPLEX_DOUBLE_EQ(val1, val2) \
  EXPECT_THAT(val1, ExpectComplexDoubleEq(val2))

// Bit-for-bit, unlike the FloatEq above: the COMPLEX(4) power tests below
// check the exact result, not a result within a few ULPs of it. Eq() alone
// would not do -- it is numeric equality, so it cannot tell +0.0f from
// -0.0f -- hence the explicit bit-pattern comparison.
MATCHER_P(ExpectComplexFloatExactlyEq, c, "") {
  auto sameBits = [](float a, float b) {
    return std::memcmp(&a, &b, sizeof(float)) == 0;
  };
  return sameBits(arg.real(), c.real()) && sameBits(arg.imag(), c.imag());
}

#define EXPECT_COMPLEX_FLOAT_EXACTLY_EQ(val1, val2) \
  EXPECT_THAT(val1, ExpectComplexFloatExactlyEq(val2))

using namespace std::literals::complex_literals;

TEST(Complex, cpowi) {
  EXPECT_COMPLEX_FLOAT_EQ(cpowi(3.f + 4if, 0), 1.f + 0if);
  EXPECT_COMPLEX_FLOAT_EQ(cpowi(3.f + 4if, 1), 3.f + 4if);

  EXPECT_COMPLEX_FLOAT_EQ(cpowi(3.f + 4if, 2), -7.f + 24if);
  EXPECT_COMPLEX_FLOAT_EQ(cpowi(3.f + 4if, 3), -117.f + 44if);
  EXPECT_COMPLEX_FLOAT_EQ(cpowi(3.f + 4if, 4), -527.f - 336if);

  EXPECT_COMPLEX_FLOAT_EQ(cpowi(3.f + 4if, -2), -0.0112f - 0.0384if);
  EXPECT_COMPLEX_FLOAT_EQ(cpowi(2.f + 1if, 10), -237.f - 3116if);
  EXPECT_COMPLEX_FLOAT_EQ(cpowi(0.5f + 0.6if, -10), -9.322937f - 7.2984829if);

  EXPECT_COMPLEX_FLOAT_EQ(cpowi(2.f + 1if, 5), -38.f + 41if);
  EXPECT_COMPLEX_FLOAT_EQ(cpowi(0.5f + 0.6if, -5), -1.121837f + 3.252915if);

  EXPECT_COMPLEX_FLOAT_EQ(
      cpowi(0.f + 1if, std::numeric_limits<std::int32_t>::min()), 1.f + 0if);
}

TEST(Complex, cpowk) {
  EXPECT_COMPLEX_FLOAT_EQ(cpowk(3.f + 4if, 0), 1.f + 0if);
  EXPECT_COMPLEX_FLOAT_EQ(cpowk(3.f + 4if, 1), 3.f + 4if);
  EXPECT_COMPLEX_FLOAT_EQ(cpowk(3.f + 4if, 2), -7.f + 24if);
  EXPECT_COMPLEX_FLOAT_EQ(cpowk(3.f + 4if, 3), -117.f + 44if);
  EXPECT_COMPLEX_FLOAT_EQ(cpowk(3.f + 4if, 4), -527.f - 336if);

  EXPECT_COMPLEX_FLOAT_EQ(cpowk(3.f + 4if, -2), -0.0112f - 0.0384if);
  EXPECT_COMPLEX_FLOAT_EQ(cpowk(2.f + 1if, 10), -237.f - 3116if);
  EXPECT_COMPLEX_FLOAT_EQ(cpowk(0.5f + 0.6if, -10), -9.322937f - 7.2984829if);

  EXPECT_COMPLEX_FLOAT_EQ(cpowk(2.f + 1if, 5), -38.f + 41if);
  EXPECT_COMPLEX_FLOAT_EQ(cpowk(0.5f + 0.6if, -5), -1.121837f + 3.252915if);

  EXPECT_COMPLEX_FLOAT_EQ(
      cpowk(0.f + 1if, std::numeric_limits<std::int64_t>::min()), 1.f + 0if);
}

TEST(Complex, zpowi) {
  EXPECT_COMPLEX_DOUBLE_EQ(zpowi(3. + 4i, 0), 1. + 0i);
  EXPECT_COMPLEX_DOUBLE_EQ(zpowi(3. + 4i, 1), 3. + 4i);
  EXPECT_COMPLEX_DOUBLE_EQ(zpowi(3. + 4i, 2), -7. + 24i);
  EXPECT_COMPLEX_DOUBLE_EQ(zpowi(3. + 4i, 3), -117. + 44i);
  EXPECT_COMPLEX_DOUBLE_EQ(zpowi(3. + 4i, 4), -527. - 336i);

  EXPECT_COMPLEX_DOUBLE_EQ(zpowi(3. + 4i, -2), -0.0112 - 0.0384i);
  EXPECT_COMPLEX_DOUBLE_EQ(zpowi(2. + 1i, 10), -237. - 3116i);
  EXPECT_COMPLEX_DOUBLE_EQ(zpowi(0.5 + 0.6i, -10), -9.32293628 - 7.29848564i);

  EXPECT_COMPLEX_DOUBLE_EQ(zpowi(2. + 1i, 5), -38. + 41i);
  EXPECT_COMPLEX_DOUBLE_EQ(zpowi(0.5 + 0.6i, -5), -1.12183773 + 3.25291503i);

  EXPECT_COMPLEX_DOUBLE_EQ(
      zpowi(0. + 1i, std::numeric_limits<std::int32_t>::min()), 1. + 0i);
}

TEST(Complex, zpowk) {
  EXPECT_COMPLEX_DOUBLE_EQ(zpowk(3. + 4i, 0), 1. + 0i);
  EXPECT_COMPLEX_DOUBLE_EQ(zpowk(3. + 4i, 1), 3. + 4i);
  EXPECT_COMPLEX_DOUBLE_EQ(zpowk(3. + 4i, 2), -7. + 24i);
  EXPECT_COMPLEX_DOUBLE_EQ(zpowk(3. + 4i, 3), -117. + 44i);
  EXPECT_COMPLEX_DOUBLE_EQ(zpowk(3. + 4i, 4), -527. - 336i);

  EXPECT_COMPLEX_DOUBLE_EQ(zpowk(3. + 4i, -2), -0.0112 - 0.0384i);
  EXPECT_COMPLEX_DOUBLE_EQ(zpowk(2. + 1i, 10), -237. - 3116i);
  EXPECT_COMPLEX_DOUBLE_EQ(zpowk(0.5 + 0.6i, -10), -9.32293628 - 7.29848564i);

  EXPECT_COMPLEX_DOUBLE_EQ(zpowk(2. + 1i, 5l), -38. + 41i);
  EXPECT_COMPLEX_DOUBLE_EQ(zpowk(0.5 + 0.6i, -5), -1.12183773 + 3.25291503i);

  EXPECT_COMPLEX_DOUBLE_EQ(
      zpowk(0. + 1i, std::numeric_limits<std::int64_t>::min()), 1. + 0i);
}

// Exact bit-pattern check, unlike FloatEq above: folded constants and
// runtime evaluation must agree bit for bit (complex-powi.cpp). Reference
// values are the double-precision powers rounded once to single; each
// differs from what per-step single-precision rounding produces.
TEST(Complex, cpowiRoundsToSingleOnce) {
  EXPECT_COMPLEX_FLOAT_EXACTLY_EQ(cpowi(1.234567f + 1.234567if, 7),
      0x1.17c216p+5f - 0x1.17c216p+5if); // (34.9697685, -34.9697685)
  EXPECT_COMPLEX_FLOAT_EXACTLY_EQ(cpowi(0.5f + 0.6if, -10),
      -0x1.2a557cp+3f - 0x1.d31a54p+2if); // (-9.3229351, -7.29848194)
  EXPECT_COMPLEX_FLOAT_EXACTLY_EQ(cpowi(0.5f + 0.6if, -5),
      -0x1.1f30bap+0f + 0x1.a05f82p+1if); // (-1.12183726, 3.25291467)
  EXPECT_COMPLEX_FLOAT_EXACTLY_EQ(cpowi(1.1f - 0.7if, 13),
      0x1.d6d9e8p+3f - 0x1.bd1ecap+4if); // (14.7140999, -27.8200169)
  EXPECT_COMPLEX_FLOAT_EXACTLY_EQ(cpowi(1.0625f + 0.8125if, 21),
      0x1.747dfap+7f + 0x1.98ea74p+8if); // (186.246048, 408.915833)
  EXPECT_COMPLEX_FLOAT_EXACTLY_EQ(cpowi(-1.75f + 0.375if, -7),
      -0x1.9e870cp-10f - 0x1.15590ap-6if); // (-0.00158129702, -0.0169279668)
}

// The exponent's kind does not select the accumulation precision -- the
// COMPLEX(4) result type does -- so cpowk must produce exactly the values
// above, not merely values close to them.
TEST(Complex, cpowkRoundsToSingleOnce) {
  EXPECT_COMPLEX_FLOAT_EXACTLY_EQ(
      cpowk(1.234567f + 1.234567if, 7), 0x1.17c216p+5f - 0x1.17c216p+5if);
  EXPECT_COMPLEX_FLOAT_EXACTLY_EQ(
      cpowk(0.5f + 0.6if, -10), -0x1.2a557cp+3f - 0x1.d31a54p+2if);
  EXPECT_COMPLEX_FLOAT_EXACTLY_EQ(
      cpowk(0.5f + 0.6if, -5), -0x1.1f30bap+0f + 0x1.a05f82p+1if);
  EXPECT_COMPLEX_FLOAT_EXACTLY_EQ(
      cpowk(1.1f - 0.7if, 13), 0x1.d6d9e8p+3f - 0x1.bd1ecap+4if);
  EXPECT_COMPLEX_FLOAT_EXACTLY_EQ(
      cpowk(1.0625f + 0.8125if, 21), 0x1.747dfap+7f + 0x1.98ea74p+8if);
  EXPECT_COMPLEX_FLOAT_EXACTLY_EQ(
      cpowk(-1.75f + 0.375if, -7), -0x1.9e870cp-10f - 0x1.15590ap-6if);
}
