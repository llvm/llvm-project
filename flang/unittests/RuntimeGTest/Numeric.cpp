//===-- flang/unittests/RuntimeGTest/Numeric.cpp ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../../runtime/numeric.h"
#include "gtest/gtest.h"
#include <cmath>
#include <limits>

using namespace Fortran::runtime;
using Fortran::common::TypeCategory;
template <int KIND> using Int = CppTypeFor<TypeCategory::Integer, KIND>;
template <int KIND> using Real = CppTypeFor<TypeCategory::Real, KIND>;

// Simple tests of numeric intrinsic functions using examples from Fortran 2018

TEST(Numeric, Aint) {
  EXPECT_EQ(RTNAME(Aint4_4)(Real<4>{3.7}), 3.0);
  EXPECT_EQ(RTNAME(Aint8_4)(Real<8>{-3.7}), -3.0);
  EXPECT_EQ(RTNAME(Aint8_8)(Real<8>{0}), 0.0);
  EXPECT_EQ(RTNAME(Aint4_4)(std::numeric_limits<Real<4>>::infinity()),
      std::numeric_limits<Real<4>>::infinity());
  EXPECT_TRUE(
      std::isnan(RTNAME(Aint8_8)(std::numeric_limits<Real<8>>::quiet_NaN())));
}

TEST(Numeric, Anint) {
  EXPECT_EQ(RTNAME(Anint4_4)(Real<4>{2.783}), 3.0);
  EXPECT_EQ(RTNAME(Anint8_4)(Real<8>{-2.783}), -3.0);
  EXPECT_EQ(RTNAME(Anint4_4)(Real<4>{2.5}), 3.0);
  EXPECT_EQ(RTNAME(Anint8_4)(Real<8>{-2.5}), -3.0);
  EXPECT_EQ(RTNAME(Anint8_8)(Real<8>{0}), 0.0);
  EXPECT_EQ(RTNAME(Anint4_4)(std::numeric_limits<Real<4>>::infinity()),
      std::numeric_limits<Real<4>>::infinity());
  EXPECT_TRUE(
      std::isnan(RTNAME(Aint8_8)(std::numeric_limits<Real<8>>::quiet_NaN())));
}

TEST(Numeric, Ceiling) {
  EXPECT_EQ(RTNAME(Ceiling4_4)(Real<4>{3.7}), 4);
  EXPECT_EQ(RTNAME(Ceiling8_8)(Real<8>{-3.7}), -3);
  EXPECT_EQ(RTNAME(Ceiling4_1)(Real<4>{0}), 0);
  EXPECT_EQ(RTNAME(Ceiling4_4)(std::numeric_limits<Real<4>>::infinity()),
      std::numeric_limits<Int<4>>::min());
  EXPECT_EQ(RTNAME(Ceiling4_4)(std::numeric_limits<Real<4>>::quiet_NaN()),
      std::numeric_limits<Int<4>>::min());
}

TEST(Numeric, Floor) {
  EXPECT_EQ(RTNAME(Floor4_4)(Real<4>{3.7}), 3);
  EXPECT_EQ(RTNAME(Floor8_8)(Real<8>{-3.7}), -4);
  EXPECT_EQ(RTNAME(Floor4_1)(Real<4>{0}), 0);
  EXPECT_EQ(RTNAME(Floor4_4)(std::numeric_limits<Real<4>>::infinity()),
      std::numeric_limits<Int<4>>::min());
  EXPECT_EQ(RTNAME(Floor4_4)(std::numeric_limits<Real<4>>::quiet_NaN()),
      std::numeric_limits<Int<4>>::min());
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
  auto nintInf{RTNAME(Nint4_4)(std::numeric_limits<Real<4>>::infinity())};
  EXPECT_TRUE(nintInf == std::numeric_limits<Int<4>>::min() ||
      nintInf == std::numeric_limits<Int<4>>::max());
  auto nintNaN{RTNAME(Nint4_4)(std::numeric_limits<Real<4>>::quiet_NaN())};
  EXPECT_TRUE(nintNaN == std::numeric_limits<Int<4>>::min() ||
      nintNaN == std::numeric_limits<Int<4>>::max());
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
  EXPECT_EQ(RTNAME(SetExponent8)(Real<8>{3.0}, 0), 1.5);
  EXPECT_EQ(RTNAME(SetExponent4)(Real<4>{1.0}, 0), 1.0);
  EXPECT_EQ(RTNAME(SetExponent4)(Real<4>{1.0}, 1), 2.0);
  EXPECT_EQ(RTNAME(SetExponent4)(Real<4>{1.0}, -1), 0.5);
  EXPECT_TRUE(std::isnan(
      RTNAME(SetExponent4)(std::numeric_limits<Real<4>>::infinity(), 1)));
  EXPECT_TRUE(std::isnan(
      RTNAME(SetExponent8)(std::numeric_limits<Real<8>>::quiet_NaN(), 1)));
}

TEST(Numeric, Spacing) {
  EXPECT_EQ(RTNAME(Spacing8)(Real<8>{0}), std::numeric_limits<Real<8>>::min());
  EXPECT_EQ(RTNAME(Spacing4)(Real<4>{3.0}), std::ldexp(Real<4>{1.0}, -22));
  EXPECT_TRUE(
      std::isnan(RTNAME(Spacing4)(std::numeric_limits<Real<4>>::infinity())));
  EXPECT_TRUE(
      std::isnan(RTNAME(Spacing8)(std::numeric_limits<Real<8>>::quiet_NaN())));
}
