//===-- flang/unittests/Runtime/Complex.cpp ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "gmock/gmock.h"
#include "gtest/gtest-matchers.h"
#include <limits>

#ifdef __clang__
#pragma clang diagnostic ignored "-Wc99-extensions"
#endif

#include "flang/Common/Fortran.h"
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
