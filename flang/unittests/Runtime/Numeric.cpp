//===-- flang/unittests/Runtime/Numeric.cpp ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/numeric.h"
#include "gtest/gtest.h"
#include <cmath>
#include <limits>

using namespace Fortran::runtime;
using Fortran::common::TypeCategory;
template <int KIND> using Int = CppTypeFor<TypeCategory::Integer, KIND>;
template <int KIND> using Real = CppTypeFor<TypeCategory::Real, KIND>;

// Simple tests of numeric intrinsic functions using examples from Fortran 2018

TEST(Numeric, Ceiling) {
  EXPECT_EQ(RTNAME(Ceiling4_4)(Real<4>{3.7}), 4);
  EXPECT_EQ(RTNAME(Ceiling8_8)(Real<8>{-3.7}), -3);
  EXPECT_EQ(RTNAME(Ceiling4_1)(Real<4>{0}), 0);
}

TEST(Numeric, Floor) {
  EXPECT_EQ(RTNAME(Floor4_4)(Real<4>{3.7}), 3);
  EXPECT_EQ(RTNAME(Floor8_8)(Real<8>{-3.7}), -4);
  EXPECT_EQ(RTNAME(Floor4_1)(Real<4>{0}), 0);
}

TEST(Numeric, Exponent) {
  EXPECT_EQ(RTNAME(Exponent4_4)(Real<4>{0}), 0);
  EXPECT_EQ(RTNAME(Exponent4_8)(Real<4>{1.0}), 1);
  EXPECT_EQ(RTNAME(Exponent8_4)(Real<8>{4.1}), 3);
  EXPECT_EQ(RTNAME(Exponent8_8)(std::numeric_limits<Real<8>>::infinity()),
      std::numeric_limits<Int<8>>::max());
  EXPECT_EQ(RTNAME(Exponent8_8)(std::numeric_limits<Real<8>>::quiet_NaN()),
      std::numeric_limits<Int<8>>::max());
}

TEST(Numeric, Fraction) {
  EXPECT_EQ(RTNAME(Fraction4)(Real<4>{0}), 0);
  EXPECT_EQ(RTNAME(Fraction8)(Real<8>{3.0}), 0.75);
  EXPECT_TRUE(
      std::isnan(RTNAME(Fraction4)(std::numeric_limits<Real<4>>::infinity())));
  EXPECT_TRUE(
      std::isnan(RTNAME(Fraction8)(std::numeric_limits<Real<8>>::quiet_NaN())));
}

TEST(Numeric, IsNaN) {
  EXPECT_FALSE(RTNAME(IsNaN4)(Real<4>{0}));
  EXPECT_FALSE(RTNAME(IsNaN8)(std::numeric_limits<Real<8>>::infinity()));
  EXPECT_TRUE(RTNAME(IsNaN8)(std::numeric_limits<Real<8>>::quiet_NaN()));
}

TEST(Numeric, Mod) {
  EXPECT_EQ(RTNAME(ModInteger1)(Int<1>{8}, Int<1>(5)), 3);
  EXPECT_EQ(RTNAME(ModInteger4)(Int<4>{-8}, Int<4>(5)), -3);
  EXPECT_EQ(RTNAME(ModInteger2)(Int<2>{8}, Int<2>(-5)), 3);
  EXPECT_EQ(RTNAME(ModInteger8)(Int<8>{-8}, Int<8>(-5)), -3);
  EXPECT_EQ(RTNAME(ModReal4)(Real<4>{8.0}, Real<4>(5.0)), 3.0);
  EXPECT_EQ(RTNAME(ModReal4)(Real<4>{-8.0}, Real<4>(5.0)), -3.0);
  EXPECT_EQ(RTNAME(ModReal8)(Real<8>{8.0}, Real<8>(-5.0)), 3.0);
  EXPECT_EQ(RTNAME(ModReal8)(Real<8>{-8.0}, Real<8>(-5.0)), -3.0);
}

TEST(Numeric, Modulo) {
  EXPECT_EQ(RTNAME(ModuloInteger1)(Int<1>{8}, Int<1>(5)), 3);
  EXPECT_EQ(RTNAME(ModuloInteger4)(Int<4>{-8}, Int<4>(5)), 2);
  EXPECT_EQ(RTNAME(ModuloInteger2)(Int<2>{8}, Int<2>(-5)), -2);
  EXPECT_EQ(RTNAME(ModuloInteger8)(Int<8>{-8}, Int<8>(-5)), -3);
  EXPECT_EQ(RTNAME(ModuloReal4)(Real<4>{8.0}, Real<4>(5.0)), 3.0);
  EXPECT_EQ(RTNAME(ModuloReal4)(Real<4>{-8.0}, Real<4>(5.0)), 2.0);
  EXPECT_EQ(RTNAME(ModuloReal8)(Real<8>{8.0}, Real<8>(-5.0)), -2.0);
  EXPECT_EQ(RTNAME(ModuloReal8)(Real<8>{-8.0}, Real<8>(-5.0)), -3.0);
}

TEST(Numeric, Nearest) {
  EXPECT_EQ(RTNAME(Nearest4)(Real<4>{0}, true),
      std::numeric_limits<Real<4>>::denorm_min());
  EXPECT_EQ(RTNAME(Nearest4)(Real<4>{3.0}, true),
      Real<4>{3.0} + std::ldexp(Real<4>{1.0}, -22));
  EXPECT_EQ(RTNAME(Nearest8)(Real<8>{1.0}, true),
      Real<8>{1.0} + std::ldexp(Real<8>{1.0}, -52));
  EXPECT_EQ(RTNAME(Nearest8)(Real<8>{1.0}, false),
      Real<8>{1.0} - std::ldexp(Real<8>{1.0}, -52));
}

TEST(Numeric, Nint) {
  EXPECT_EQ(RTNAME(Nint4_4)(Real<4>{2.783}), 3);
  EXPECT_EQ(RTNAME(Nint8_4)(Real<8>{-2.783}), -3);
  EXPECT_EQ(RTNAME(Nint4_4)(Real<4>{2.5}), 3);
  EXPECT_EQ(RTNAME(Nint8_4)(Real<8>{-2.5}), -3);
  EXPECT_EQ(RTNAME(Nint8_8)(Real<8>{0}), 0);
}

TEST(Numeric, RRSpacing) {
  EXPECT_EQ(RTNAME(RRSpacing8)(Real<8>{0}), 0);
  EXPECT_EQ(RTNAME(RRSpacing4)(Real<4>{-3.0}), 0.75 * (1 << 24));
  EXPECT_EQ(RTNAME(RRSpacing8)(Real<8>{-3.0}), 0.75 * (std::int64_t{1} << 53));
  EXPECT_TRUE(
      std::isnan(RTNAME(RRSpacing4)(std::numeric_limits<Real<4>>::infinity())));
  EXPECT_TRUE(std::isnan(
      RTNAME(RRSpacing8)(std::numeric_limits<Real<8>>::quiet_NaN())));
}

TEST(Numeric, Scale) {
  EXPECT_EQ(RTNAME(Scale4)(Real<4>{0}, 0), 0);
  EXPECT_EQ(RTNAME(Scale4)(Real<4>{1.0}, 0), 1.0);
  EXPECT_EQ(RTNAME(Scale4)(Real<4>{1.0}, 1), 2.0);
  EXPECT_EQ(RTNAME(Scale4)(Real<4>{1.0}, -1), 0.5);
  EXPECT_TRUE(
      std::isinf(RTNAME(Scale4)(std::numeric_limits<Real<4>>::infinity(), 1)));
  EXPECT_TRUE(
      std::isnan(RTNAME(Scale8)(std::numeric_limits<Real<8>>::quiet_NaN(), 1)));
}

TEST(Numeric, SetExponent) {
  EXPECT_EQ(RTNAME(SetExponent4)(Real<4>{0}, 0), 0);
  EXPECT_EQ(RTNAME(SetExponent8)(Real<8>{0}, 666), 0);
  EXPECT_EQ(RTNAME(SetExponent8)(Real<8>{3.0}, 0), 0.75);
  EXPECT_EQ(RTNAME(SetExponent4)(Real<4>{1.0}, 0), 0.5);
  EXPECT_EQ(RTNAME(SetExponent4)(Real<4>{1.0}, 1), 1.0);
  EXPECT_EQ(RTNAME(SetExponent4)(Real<4>{1.0}, -1), 0.25);
  EXPECT_TRUE(std::isnan(
      RTNAME(SetExponent4)(std::numeric_limits<Real<4>>::infinity(), 1)));
  EXPECT_TRUE(std::isnan(
      RTNAME(SetExponent8)(std::numeric_limits<Real<8>>::quiet_NaN(), 1)));
}

TEST(Numeric, SelectedIntKind) {
  std::int8_t r0 = 1;
  std::int16_t r1 = 3;
  std::int32_t r2 = 8;
  std::int64_t r3 = 10;
  std::int32_t r4 = -10;
  std::int32_t r5 = 100;
  EXPECT_EQ(RTNAME(SelectedIntKind)(__FILE__, __LINE__, &r0, 1), 1);
  EXPECT_EQ(RTNAME(SelectedIntKind)(__FILE__, __LINE__, &r1, 2), 2);
  EXPECT_EQ(RTNAME(SelectedIntKind)(__FILE__, __LINE__, &r2, 4), 4);
  EXPECT_EQ(RTNAME(SelectedIntKind)(__FILE__, __LINE__, &r3, 8), 8);
  EXPECT_EQ(RTNAME(SelectedIntKind)(__FILE__, __LINE__, &r4, 4), 1);
  EXPECT_EQ(RTNAME(SelectedIntKind)(__FILE__, __LINE__, &r5, 4), -1);
}

TEST(Numeric, SelectedRealKind) {
  std::int8_t p_s = 1;
  std::int16_t p[11] = {-10, 1, 1, 4, 50, 1, 1, 4, 1, 1, 50};
  std::int32_t r[11] = {-1, 1, 1, 1, 2, 1, 20, 20, 100, 5000, 5000};
  std::int64_t d[11] = {2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2};
  EXPECT_EQ(RTNAME(SelectedRealKind)(
                __FILE__, __LINE__, &p[0], 2, &r[0], 4, &d[0], 8),
      2);
  EXPECT_EQ(RTNAME(SelectedRealKind)(
                __FILE__, __LINE__, &p[1], 2, &r[1], 4, &d[1], 8),
      -5);
  EXPECT_EQ(RTNAME(SelectedRealKind)(
                __FILE__, __LINE__, &p[2], 2, &r[2], 4, &d[2], 8),
      2);
  EXPECT_EQ(RTNAME(SelectedRealKind)(
                __FILE__, __LINE__, &p[3], 2, &r[3], 4, &d[3], 8),
      4);
  EXPECT_EQ(RTNAME(SelectedRealKind)(
                __FILE__, __LINE__, &p[4], 2, &r[4], 4, &d[4], 8),
      -1);
  EXPECT_EQ(RTNAME(SelectedRealKind)(
                __FILE__, __LINE__, &p[5], 2, &r[5], 4, &d[5], 8),
      2);
  EXPECT_EQ(RTNAME(SelectedRealKind)(
                __FILE__, __LINE__, &p[6], 2, &r[6], 4, &d[6], 8),
      3);
  EXPECT_EQ(RTNAME(SelectedRealKind)(
                __FILE__, __LINE__, &p[7], 2, &r[7], 4, &d[7], 8),
      4);
  EXPECT_EQ(RTNAME(SelectedRealKind)(
                __FILE__, __LINE__, &p[8], 2, &r[8], 4, &d[8], 8),
      8);
  EXPECT_EQ(RTNAME(SelectedRealKind)(
                __FILE__, __LINE__, &p[9], 2, &r[9], 4, &d[9], 8),
      -2);
  EXPECT_EQ(RTNAME(SelectedRealKind)(
                __FILE__, __LINE__, &p[10], 2, &r[10], 4, &d[10], 8),
      -3);
  EXPECT_EQ(
      RTNAME(SelectedRealKind)(__FILE__, __LINE__, &p_s, 1, &r[0], 4, &d[0], 8),
      2);
  EXPECT_EQ(RTNAME(SelectedRealKind)(
                __FILE__, __LINE__, nullptr, 0, &r[0], 4, &d[0], 8),
      2);
  EXPECT_EQ(RTNAME(SelectedRealKind)(
                __FILE__, __LINE__, &p[0], 2, nullptr, 0, &d[0], 8),
      2);
  EXPECT_EQ(RTNAME(SelectedRealKind)(
                __FILE__, __LINE__, &p[0], 2, &r[0], 4, nullptr, 0),
      2);
}

TEST(Numeric, Spacing) {
  EXPECT_EQ(RTNAME(Spacing8)(Real<8>{0}), std::numeric_limits<Real<8>>::min());
  EXPECT_EQ(RTNAME(Spacing4)(Real<4>{3.0}), std::ldexp(Real<4>{1.0}, -22));
  EXPECT_TRUE(
      std::isnan(RTNAME(Spacing4)(std::numeric_limits<Real<4>>::infinity())));
  EXPECT_TRUE(
      std::isnan(RTNAME(Spacing8)(std::numeric_limits<Real<8>>::quiet_NaN())));
}
