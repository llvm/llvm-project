//===-- flang/unittests/Evaluate/RealValueTest.cpp ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "flang/Common/Fortran-consts.h"
#include "flang/Common/type-kinds.h"
#include "flang/Evaluate/integer-value.h"
#include "flang/Evaluate/real-value.h"
#include "flang/Evaluate/typekind-traits.h"
#include "llvm/Support/raw_ostream.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string>

using namespace Fortran::common;
using namespace Fortran::evaluate;
using namespace Fortran::evaluate::value;

namespace {

//===----------------------------------------------------------------------===//
// Parameterization over the REAL kinds
//===----------------------------------------------------------------------===//

struct KindName {
  template <typename TP> static std::string GetName(int) {
    return "REAL_" + std::to_string(TP::kind);
  }
};

// The subset of REAL kinds with a portable native host arithmetic type
// (float for REAL(4), double for REAL(8)), used to cross-check against
// hardware arithmetic.
using RealHostTypedKinds = testing::Types<TypeKind<TypeCategory::Real, 4>,
    TypeKind<TypeCategory::Real, 8>>;

template <typename T> class RealValueHostTypedKind : public testing::Test {};
TYPED_TEST_SUITE(RealValueHostTypedKind, RealHostTypedKinds, KindName);

class RealValueKind : public testing::TestWithParam<int> {};
INSTANTIATE_TEST_SUITE_P(RealValueKind, RealValueKind,
    testing::ValuesIn(KindsByType<TypeCategory::Real>::kinds),
    [](const testing::TestParamInfo<int> &info) {
      return "REAL_" + std::to_string(info.param);
    });

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

testing::AssertionResult RealValuesEqual(const char *lhsExpr,
    const char *rhsExpr, const RealValue &lhs, const RealValue &rhs) {
  if (lhs == rhs) {
    return testing::AssertionSuccess();
  }
  return testing::AssertionFailure()
      << lhsExpr << " (" << lhs.DumpHexadecimal() << ") != " << rhsExpr << " ("
      << rhs.DumpHexadecimal() << ")";
}

#define EXPECT_REAL_EQ(lhs, rhs) EXPECT_PRED_FORMAT2(RealValuesEqual, lhs, rhs)

std::string AsFortranString(const RealValue &x, int kind, bool minimal) {
  std::string s;
  llvm::raw_string_ostream os{s};
  x.AsFortran(os, kind, minimal);
  return s;
}

/// Takes an integer and distributes its bits across a floating-point value so
/// that a short sweep still covers signs, zeroes, subnormals, infinities and
/// NaNs. The LSB complements the result. Copied from the legacy real.cpp.
static std::uint32_t SpreadBits(std::uint32_t n) {
  static const int shifts[]{
      -1, 31, 23, 30, 22, 0, 24, 29, 25, 28, 26, 1, 16, 21, 2, -1};
  std::uint32_t x{0};
  for (int j{1}; shifts[j] >= 0; ++j) {
    x |= ((n >> j) & 1) << shifts[j];
  }
  x ^= -(n & 1);
  return x;
}

static std::uint64_t SpreadBits(std::uint64_t n) {
  static const int shifts[]{
      -1, 63, 52, 62, 51, 0, 53, 61, 54, 60, 55, 59, 1, 16, 50, 2, -1};
  std::uint64_t x{0};
  for (int j{1}; shifts[j] >= 0; ++j) {
    x |= ((n >> j) & 1) << shifts[j];
  }
  x ^= -(n & 1);
  return x;
}

/// Compares a computed RealValue against the result the host produced for the
/// same operation.  NaN payloads are not part of the contract, so only the
/// NaN-ness is compared for those.
template <typename HostT, typename UnsignedT>
static void ExpectSameAsHost(const RealValue &got, HostT expected) {
  if (std::isnan(expected)) {
    EXPECT_TRUE(got.IsNotANumber())
        << "expected NaN, got " << got.DumpHexadecimal();
    return;
  }
  union {
    UnsignedT ui;
    HostT f;
  } u;
  u.f = expected;
  EXPECT_EQ(std::uint64_t{u.ui}, got.RawBits().ToUInt64())
      << "expected " << double{expected} << ", got " << got.DumpHexadecimal();
}

static constexpr int KindPos(int kind) {
  constexpr auto &RealKinds{KindsByType<TypeCategory::Real>::kinds};
  for (std::size_t i{0}; i < std::size(RealKinds); ++i) {
    if (RealKinds[i] == kind) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

//===----------------------------------------------------------------------===//
// Construction, assignment and kind inquiries
//===----------------------------------------------------------------------===//

TEST(RealValue, DefaultConstructionIsMonostate) {
  RealValue x;
  EXPECT_TRUE(x.IsMonostate());
  EXPECT_TRUE(x.IsZero());
  EXPECT_FALSE(x.IsNegative());
  EXPECT_FALSE(x.IsNotANumber());
  EXPECT_FALSE(x.IsSignalingNaN());
  EXPECT_FALSE(x.IsInfinite());
  EXPECT_TRUE(x.IsFinite());
  EXPECT_TRUE(x.IsNormal());
  EXPECT_EQ(0, x.Exponent());
  EXPECT_TRUE(x.RawBits().IsZero());
}

TEST_P(RealValueKind, ConstructFromWord) {
  const int kind{GetParam()};
  // The word is the raw bit pattern, not a numeric value.
  RealValue zero{kind, IntegerValue::Zero(kind)};
  EXPECT_EQ(kind, zero.kind());
  EXPECT_TRUE(zero.IsZero());
  EXPECT_FALSE(zero.IsNegative());
  RealValue minusZero{RealValue::NegativeZero(kind)};
  EXPECT_TRUE(minusZero.IsZero());
  EXPECT_TRUE(minusZero.IsNegative());
}

TEST(RealValue, CopyAndMove) {
  RealValue x{4, 3.0};
  RealValue copyConstructed{x};
  EXPECT_REAL_EQ(x, copyConstructed);
  RealValue copyAssigned;
  copyAssigned = x;
  EXPECT_REAL_EQ(x, copyAssigned);
  RealValue moveConstructed{std::move(copyConstructed)};
  EXPECT_REAL_EQ(x, moveConstructed);
  RealValue moveAssigned;
  moveAssigned = std::move(copyAssigned);
  EXPECT_REAL_EQ(x, moveAssigned);
}

TEST(RealValue, KindCheckingConstructors) {
  RealValue x{4, 3.0};
  EXPECT_EQ(4, RealValue(4, x).kind());
  EXPECT_REAL_EQ(x, RealValue(4, x));
  RealValue y{8, 3.0};
  RealValue moved{8, std::move(y)};
  EXPECT_EQ(8, moved.kind());
}

TEST_P(RealValueKind, Zero) {
  const int kind{GetParam()};
  RealValue zero{RealValue::Zero(kind)};
  EXPECT_FALSE(zero.IsMonostate());
  EXPECT_EQ(kind, zero.kind());
  EXPECT_TRUE(zero.IsZero());
  EXPECT_FALSE(zero.IsNegative());
  EXPECT_TRUE(zero.RawBits().IsZero());
  EXPECT_EQ(0, zero.Exponent());
  EXPECT_EQ(Relation::Equal, zero.Compare(zero));
}

TEST(RealValue, Bits) {
  EXPECT_EQ(16, RealValue::bits(2));
  EXPECT_EQ(16, RealValue::bits(3));
  EXPECT_EQ(32, RealValue::bits(4));
  EXPECT_EQ(64, RealValue::bits(8));
  EXPECT_EQ(128, RealValue::bits(10)); // 80 significant bits, 128 stored
  EXPECT_EQ(128, RealValue::bits(16));
  EXPECT_EQ(32, (RealValue{4, 1.0}.bits()));
}

TEST(RealValue, BytesStored) {
  EXPECT_EQ(2u, RealValue::bytesStored(2));
  EXPECT_EQ(2u, RealValue::bytesStored(3));
  EXPECT_EQ(4u, RealValue::bytesStored(4));
  EXPECT_EQ(8u, RealValue::bytesStored(8));
  EXPECT_EQ(16u, RealValue::bytesStored(10));
  EXPECT_EQ(16u, RealValue::bytesStored(16));
  EXPECT_EQ(4u, (RealValue{4, 1.0}.bytesStored()));
}

TEST(RealValue, KindProperties) {
  struct {
    int kind, digits, precision, range, maxExponent, minExponent;
  } expected[]{
      {2, 11, 3, 4, 16, -13},
      {3, 8, 2, 37, 128, -125},
      {4, 24, 6, 37, 128, -125},
      {8, 53, 15, 307, 1024, -1021},
      {10, 64, 18, 4931, 16384, -16381},
      {16, 113, 33, 4931, 16384, -16381},
  };
  for (auto &e : expected) {
    SCOPED_TRACE(testing::Message() << "kind=" << e.kind);
    EXPECT_EQ(e.digits, RealValue::DIGITS(e.kind));
    EXPECT_EQ(e.precision, RealValue::PRECISION(e.kind));
    EXPECT_EQ(e.range, RealValue::RANGE(e.kind));
    EXPECT_EQ(e.maxExponent, RealValue::MAXEXPONENT(e.kind));
    EXPECT_EQ(e.minExponent, RealValue::MINEXPONENT(e.kind));
  }
}

//===----------------------------------------------------------------------===//
// Classification predicates
//===----------------------------------------------------------------------===//

TEST_P(RealValueKind, IsZero) {
  const int kind{GetParam()};
  EXPECT_TRUE(RealValue::Zero(kind).IsZero());
  EXPECT_TRUE(RealValue::NegativeZero(kind).IsZero());
  EXPECT_FALSE((RealValue{kind, 1.0}.IsZero()));
  EXPECT_FALSE(RealValue::Infinity(kind).IsZero());
  EXPECT_FALSE(RealValue::NotANumber(kind).IsZero());
}

TEST_P(RealValueKind, IsNegative) {
  const int kind{GetParam()};
  EXPECT_FALSE(RealValue::Zero(kind).IsNegative());
  EXPECT_TRUE(RealValue::NegativeZero(kind).IsNegative());
  EXPECT_FALSE((RealValue{kind, 1.0}.IsNegative()));
  EXPECT_TRUE((RealValue{kind, -1.0}.IsNegative()));
  EXPECT_TRUE(RealValue::Infinity(kind, /*negative=*/true).IsNegative());

  // A NaN is never reported as negative, whatever its sign bit.
  EXPECT_FALSE(RealValue::NotANumber(kind).IsNegative());
}

TEST_P(RealValueKind, IsNotANumber) {
  const int kind{GetParam()};
  EXPECT_FALSE(RealValue::Zero(kind).IsNotANumber());
  EXPECT_FALSE(RealValue::Infinity(kind).IsNotANumber());
  EXPECT_FALSE(RealValue::Infinity(kind, /*negative=*/true).IsNotANumber());
  EXPECT_TRUE(RealValue::NotANumber(kind).IsNotANumber());
}

TEST_P(RealValueKind, IsSignalingNaN) {
  const int kind{GetParam()};
  EXPECT_FALSE(RealValue::Zero(kind).IsSignalingNaN());
  EXPECT_FALSE(RealValue::Infinity(kind).IsSignalingNaN());

  // NotANumber() produces a quiet NaN.
  EXPECT_FALSE(RealValue::NotANumber(kind).IsSignalingNaN());
}

TEST_P(RealValueKind, IsInfinite) {
  const int kind{GetParam()};
  EXPECT_FALSE(RealValue::Zero(kind).IsInfinite());
  EXPECT_FALSE(RealValue::HUGE(kind).IsInfinite());
  EXPECT_TRUE(RealValue::Infinity(kind).IsInfinite());
  EXPECT_TRUE(RealValue::Infinity(kind, /*negative=*/true).IsInfinite());
  EXPECT_FALSE(RealValue::NotANumber(kind).IsInfinite());
}

TEST_P(RealValueKind, IsFinite) {
  const int kind{GetParam()};
  EXPECT_TRUE(RealValue::Zero(kind).IsFinite());
  EXPECT_TRUE(RealValue::HUGE(kind).IsFinite());
  EXPECT_TRUE(RealValue::TINY(kind).IsFinite());
  EXPECT_FALSE(RealValue::Infinity(kind).IsFinite());
  EXPECT_FALSE(RealValue::Infinity(kind, /*negative=*/true).IsFinite());
  EXPECT_FALSE(RealValue::NotANumber(kind).IsFinite());
}

TEST_P(RealValueKind, IsNormal) {
  const int kind{GetParam()};
  EXPECT_TRUE(RealValue::Zero(kind).IsNormal());
  EXPECT_TRUE(RealValue::TINY(kind).IsNormal());
  EXPECT_TRUE(RealValue::HUGE(kind).IsNormal());
  EXPECT_FALSE(RealValue::Infinity(kind).IsNormal());
  EXPECT_FALSE(RealValue::NotANumber(kind).IsNormal());

  // The smallest subnormal is not normal.
  RealValue subnormal{kind, IntegerValue{kind, 1}};
  EXPECT_FALSE(subnormal.IsNormal());
}

//===----------------------------------------------------------------------===//
// Sign manipulation
//===----------------------------------------------------------------------===//

TEST_P(RealValueKind, ABS) {
  const int kind{GetParam()};
  EXPECT_REAL_EQ((RealValue{kind, 3.0}), (RealValue{kind, -3.0}.ABS()));
  EXPECT_REAL_EQ((RealValue{kind, 3.0}), (RealValue{kind, 3.0}.ABS()));
  EXPECT_TRUE(RealValue::NegativeZero(kind).ABS().RawBits().IsZero());
  EXPECT_REAL_EQ(
      RealValue::Infinity(kind), RealValue::Infinity(kind, true).ABS());
}

TEST_P(RealValueKind, SetSign) {
  const int kind{GetParam()};
  EXPECT_REAL_EQ((RealValue{kind, -3.0}), (RealValue{kind, 3.0}.SetSign(true)));
  EXPECT_REAL_EQ(
      (RealValue{kind, 3.0}), (RealValue{kind, -3.0}.SetSign(false)));
  EXPECT_REAL_EQ((RealValue{kind, 3.0}), (RealValue{kind, 3.0}.SetSign(false)));
  EXPECT_REAL_EQ(
      RealValue::NegativeZero(kind), RealValue::Zero(kind).SetSign(true));
}

TEST_P(RealValueKind, SIGN) {
  const int kind{GetParam()};
  EXPECT_REAL_EQ((RealValue{kind, -3.0}),
      (RealValue{kind, 3.0}.SIGN(RealValue{kind, -1.0})));
  EXPECT_REAL_EQ((RealValue{kind, 3.0}),
      (RealValue{kind, -3.0}.SIGN(RealValue{kind, 1.0})));
  // The sign is taken from the sign bit, so -0.0 makes the result negative.
  EXPECT_REAL_EQ((RealValue{kind, -3.0}),
      (RealValue{kind, 3.0}.SIGN(RealValue::NegativeZero(kind))));
}

TEST_P(RealValueKind, Negate) {
  const int kind{GetParam()};
  EXPECT_REAL_EQ((RealValue{kind, -3.0}), (RealValue{kind, 3.0}.Negate()));
  EXPECT_REAL_EQ((RealValue{kind, 3.0}), (RealValue{kind, -3.0}.Negate()));
  EXPECT_REAL_EQ(RealValue::NegativeZero(kind), RealValue::Zero(kind).Negate());
  EXPECT_TRUE(RealValue::NegativeZero(kind).Negate().RawBits().IsZero());
  EXPECT_REAL_EQ(
      RealValue::Infinity(kind, true), RealValue::Infinity(kind).Negate());
}

//===----------------------------------------------------------------------===//
// Comparison
//===----------------------------------------------------------------------===//

TEST_P(RealValueKind, Compare) {
  const int kind{GetParam()};
  RealValue zero{RealValue::Zero(kind)};
  RealValue minusZero{RealValue::NegativeZero(kind)};
  RealValue one{kind, 1.0};
  RealValue two{kind, 2.0};
  RealValue inf{RealValue::Infinity(kind)};
  RealValue negInf{RealValue::Infinity(kind, true)};
  RealValue nan{RealValue::NotANumber(kind)};

  EXPECT_EQ(Relation::Equal, zero.Compare(zero));
  EXPECT_EQ(Relation::Equal, zero.Compare(minusZero)); // +0 == -0
  EXPECT_EQ(Relation::Equal, minusZero.Compare(minusZero));
  EXPECT_EQ(Relation::Less, one.Compare(two));
  EXPECT_EQ(Relation::Greater, two.Compare(one));
  EXPECT_EQ(Relation::Less, zero.Compare(inf));
  EXPECT_EQ(Relation::Less, minusZero.Compare(inf));
  EXPECT_EQ(Relation::Greater, zero.Compare(negInf));
  EXPECT_EQ(Relation::Greater, minusZero.Compare(negInf));
  EXPECT_EQ(Relation::Equal, inf.Compare(inf));
  EXPECT_EQ(Relation::Equal, negInf.Compare(negInf));
  EXPECT_EQ(Relation::Greater, inf.Compare(negInf));
  // Every comparison against a NaN is unordered.
  EXPECT_EQ(Relation::Unordered, nan.Compare(nan));
  EXPECT_EQ(Relation::Unordered, zero.Compare(nan));
  EXPECT_EQ(Relation::Unordered, minusZero.Compare(nan));
  EXPECT_EQ(Relation::Unordered, nan.Compare(zero));
  EXPECT_EQ(Relation::Unordered, nan.Compare(inf));
  EXPECT_EQ(Relation::Unordered, nan.Compare(negInf));
}

TEST_P(RealValueKind, EqualityOperators) {
  const int kind{GetParam()};
  // operator== compares bit patterns, unlike Compare().
  EXPECT_TRUE((RealValue{kind, 1.0} == RealValue{kind, 1.0}));
  EXPECT_FALSE((RealValue{kind, 1.0} == RealValue{kind, 2.0}));
  EXPECT_TRUE((RealValue{kind, 1.0} != RealValue{kind, 2.0}));
  EXPECT_FALSE(RealValue::Zero(kind) == RealValue::NegativeZero(kind));
  EXPECT_TRUE(RealValue::NotANumber(kind) == RealValue::NotANumber(kind));
}

//===----------------------------------------------------------------------===//
// Arithmetic
//===----------------------------------------------------------------------===//

TEST_P(RealValueKind, Add) {
  const int kind{GetParam()};
  auto sum{RealValue{kind, 1.0}.Add(RealValue{kind, 2.0})};
  EXPECT_TRUE(sum.flags.empty());
  EXPECT_REAL_EQ((RealValue{kind, 3.0}), sum.value);
  // Cancellation yields +0.0.
  auto cancelled{RealValue{kind, 3.0}.Add(RealValue{kind, -3.0})};
  EXPECT_TRUE(cancelled.value.IsZero());
  EXPECT_FALSE(cancelled.value.IsNegative());
  // Overflow.
  auto overflowed{RealValue::HUGE(kind).Add(RealValue::HUGE(kind))};
  EXPECT_TRUE(overflowed.flags.test(RealFlag::Overflow));
  EXPECT_TRUE(overflowed.value.IsInfinite());
  // Inf + (-Inf) is invalid.
  auto invalid{RealValue::Infinity(kind).Add(RealValue::Infinity(kind, true))};
  EXPECT_TRUE(invalid.flags.test(RealFlag::InvalidArgument));
  EXPECT_TRUE(invalid.value.IsNotANumber());
}

TEST_P(RealValueKind, Subtract) {
  const int kind{GetParam()};
  auto diff{RealValue{kind, 3.0}.Subtract(RealValue{kind, 5.0})};
  EXPECT_TRUE(diff.flags.empty());
  EXPECT_REAL_EQ((RealValue{kind, -2.0}), diff.value);
  auto invalid{RealValue::Infinity(kind).Subtract(RealValue::Infinity(kind))};
  EXPECT_TRUE(invalid.flags.test(RealFlag::InvalidArgument));
  EXPECT_TRUE(invalid.value.IsNotANumber());
}

TEST_P(RealValueKind, Multiply) {
  const int kind{GetParam()};
  auto product{RealValue{kind, 3.0}.Multiply(RealValue{kind, 5.0})};
  EXPECT_TRUE(product.flags.empty());
  EXPECT_REAL_EQ((RealValue{kind, 15.0}), product.value);
  RealValue negProduct{
      RealValue{kind, -3.0}.Multiply(RealValue{kind, 5.0}).value};
  EXPECT_REAL_EQ((RealValue{kind, -15.0}), negProduct);
  auto overflowed{RealValue::HUGE(kind).Multiply(RealValue{kind, 2.0})};
  EXPECT_TRUE(overflowed.flags.test(RealFlag::Overflow));
  EXPECT_TRUE(overflowed.value.IsInfinite());
  auto underflowed{RealValue::TINY(kind).Multiply(RealValue::TINY(kind))};
  EXPECT_TRUE(underflowed.flags.test(RealFlag::Underflow));
  EXPECT_TRUE(underflowed.value.IsZero());
  // 0 * Inf is invalid.
  auto invalid{RealValue::Zero(kind).Multiply(RealValue::Infinity(kind))};
  EXPECT_TRUE(invalid.flags.test(RealFlag::InvalidArgument));
  EXPECT_TRUE(invalid.value.IsNotANumber());
}

TEST_P(RealValueKind, Divide) {
  const int kind{GetParam()};
  auto quotient{RealValue{kind, 15.0}.Divide(RealValue{kind, 5.0})};
  EXPECT_TRUE(quotient.flags.empty());
  EXPECT_REAL_EQ((RealValue{kind, 3.0}), quotient.value);
  // 1/3 is inexact in every binary format.
  auto inexact{RealValue{kind, 1.0}.Divide(RealValue{kind, 3.0})};
  EXPECT_TRUE(inexact.flags.test(RealFlag::Inexact));
  // Division by zero.
  auto byZero{RealValue{kind, 1.0}.Divide(RealValue::Zero(kind))};
  EXPECT_TRUE(byZero.flags.test(RealFlag::DivideByZero));
  EXPECT_TRUE(byZero.value.IsInfinite());
  EXPECT_FALSE(byZero.value.IsNegative());
  auto negByZero{RealValue{kind, -1.0}.Divide(RealValue::Zero(kind))};
  EXPECT_TRUE(negByZero.value.IsInfinite());
  EXPECT_TRUE(negByZero.value.IsNegative());
  // 0/0 is invalid.
  auto invalid{RealValue::Zero(kind).Divide(RealValue::Zero(kind))};
  EXPECT_TRUE(invalid.flags.test(RealFlag::InvalidArgument));
  EXPECT_TRUE(invalid.value.IsNotANumber());
}

TEST_P(RealValueKind, SQRT) {
  const int kind{GetParam()};
  auto four{RealValue{kind, 4.0}.SQRT()};
  EXPECT_TRUE(four.flags.empty());
  EXPECT_REAL_EQ((RealValue{kind, 2.0}), four.value);
  EXPECT_TRUE(RealValue::Zero(kind).SQRT().value.IsZero());
  // SQRT of a negative number is invalid.
  auto invalid{RealValue{kind, -1.0}.SQRT()};
  EXPECT_TRUE(invalid.flags.test(RealFlag::InvalidArgument));
  EXPECT_TRUE(invalid.value.IsNotANumber());
  EXPECT_TRUE(RealValue::Infinity(kind).SQRT().value.IsInfinite());
}

TEST_P(RealValueKind, NEAREST) {
  const int kind{GetParam()};
  RealValue one{kind, 1.0};

  // The next value above 1.0 is 1.0+EPSILON.
  auto up{one.NEAREST(true)};
  EXPECT_REAL_EQ(one.Add(RealValue::EPSILON(kind)).value, up.value);

  auto down{one.NEAREST(false)};
  EXPECT_EQ(Relation::Less, down.value.Compare(one));
  // Stepping back up recovers 1.0 exactly.
  EXPECT_REAL_EQ(one, down.value.NEAREST(true).value);

  // Stepping down from +0.0 gives the smallest negative subnormal.
  auto belowZero{RealValue::Zero(kind).NEAREST(false)};
  EXPECT_TRUE(belowZero.value.IsNegative());
  EXPECT_FALSE(belowZero.value.IsNormal());
}

TEST_P(RealValueKind, HYPOT) {
  const int kind{GetParam()};

  auto hypot{RealValue{kind, 3.0}.HYPOT(RealValue{kind, 4.0})};
  EXPECT_REAL_EQ((RealValue{kind, 5.0}), hypot.value);

  // HYPOT avoids the overflow that squaring HUGE would produce.
  auto big{RealValue::HUGE(kind).HYPOT(RealValue::HUGE(kind))};
  EXPECT_FALSE(big.value.IsNotANumber());
}

TEST_P(RealValueKind, DIM) {
  const int kind{GetParam()};

  auto positive{RealValue{kind, 7.0}.DIM(RealValue{kind, 5.0})};
  EXPECT_REAL_EQ((RealValue{kind, 2.0}), positive.value);

  // MAX(x-y, 0) clamps at zero.
  auto clamped{RealValue{kind, 5.0}.DIM(RealValue{kind, 7.0})};
  EXPECT_TRUE(clamped.value.IsZero());

  auto invalid{RealValue::NotANumber(kind).DIM(RealValue{kind, 1.0})};
  EXPECT_TRUE(invalid.flags.test(RealFlag::InvalidArgument));
}

TEST_P(RealValueKind, MOD) {
  const int kind{GetParam()};

  // The result has the sign of the dividend.
  RealValue m1{RealValue{kind, 8.0}.MOD(RealValue{kind, 5.0}).value};
  EXPECT_REAL_EQ((RealValue{kind, 3.0}), m1);

  RealValue m2{RealValue{kind, -8.0}.MOD(RealValue{kind, 5.0}).value};
  EXPECT_REAL_EQ((RealValue{kind, -3.0}), m2);

  RealValue m3{RealValue{kind, 8.0}.MOD(RealValue{kind, -5.0}).value};
  EXPECT_REAL_EQ((RealValue{kind, 3.0}), m3);

  auto byZero{RealValue{kind, 8.0}.MOD(RealValue::Zero(kind))};
  EXPECT_TRUE(byZero.flags.test(RealFlag::DivideByZero));
  EXPECT_TRUE(byZero.value.IsNotANumber());
}

TEST_P(RealValueKind, MODULO) {
  const int kind{GetParam()};

  // The result has the sign of the divisor.
  RealValue m1{RealValue{kind, 8.0}.MODULO(RealValue{kind, 5.0}).value};
  EXPECT_REAL_EQ((RealValue{kind, 3.0}), m1);
  RealValue m2{RealValue{kind, -8.0}.MODULO(RealValue{kind, 5.0}).value};
  EXPECT_REAL_EQ((RealValue{kind, 2.0}), m2);
  RealValue m3{RealValue{kind, 8.0}.MODULO(RealValue{kind, -5.0}).value};
  EXPECT_REAL_EQ((RealValue{kind, -2.0}), m3);
  RealValue m4{RealValue{kind, -8.0}.MODULO(RealValue{kind, -5.0}).value};
  EXPECT_REAL_EQ((RealValue{kind, -3.0}), m4);
}

TEST_P(RealValueKind, KahanSummation) {
  const int kind{GetParam()};

  RealValue correction{RealValue::Zero(kind)};
  auto sum{
      RealValue{kind, 1.0}.KahanSummation(RealValue{kind, 2.0}, correction)};
  EXPECT_REAL_EQ((RealValue{kind, 3.0}), sum.value);
  EXPECT_TRUE(correction.IsZero());
  // Adding a value too small to be representable in the sum leaves it in the
  // correction term instead of losing it.
  RealValue one{kind, 1.0};
  RealValue small{RealValue::EPSILON(kind).Divide(RealValue{kind, 4.0}).value};
  correction = RealValue::Zero(kind);
  auto lossy{one.KahanSummation(small, correction)};
  EXPECT_REAL_EQ(one, lossy.value);
  EXPECT_FALSE(correction.IsZero());
}

//===----------------------------------------------------------------------===//
// Kind-specific constants and exponent manipulation
//===----------------------------------------------------------------------===//

TEST_P(RealValueKind, EPSILON) {
  const int kind{GetParam()};

  RealValue eps{RealValue::EPSILON(kind)};
  EXPECT_EQ(kind, eps.kind());
  EXPECT_FALSE(eps.IsNegative());
  // EPSILON is the spacing of 1.0, i.e. 2**(1-DIGITS).
  EXPECT_REAL_EQ(eps, RealValue(kind, 1.0).SPACING());
  // 1+EPSILON is distinguishable from 1, but 1+EPSILON/2 is not.
  RealValue one{kind, 1.0};
  EXPECT_EQ(Relation::Greater, one.Add(eps).value.Compare(one));
  RealValue halfEps{eps.Divide(RealValue{kind, 2.0}).value};
  EXPECT_EQ(Relation::Equal, one.Add(halfEps).value.Compare(one));
}

TEST_P(RealValueKind, HUGE) {
  const int kind{GetParam()};

  RealValue huge{RealValue::HUGE(kind)};
  EXPECT_EQ(kind, huge.kind());
  EXPECT_TRUE(huge.IsFinite());
  EXPECT_FALSE(huge.IsNegative());
  // The exponent field is one below the reserved all-ones value that
  // Infinity() uses.
  EXPECT_EQ(RealValue::Infinity(kind).Exponent() - 1, huge.Exponent());
  // Stepping up from HUGE overflows to infinity.
  EXPECT_TRUE(huge.NEAREST(true).value.IsInfinite());
}

TEST_P(RealValueKind, TINY) {
  const int kind{GetParam()};

  RealValue tiny{RealValue::TINY(kind)};
  EXPECT_EQ(kind, tiny.kind());
  EXPECT_TRUE(tiny.IsNormal());
  EXPECT_FALSE(tiny.IsZero());
  EXPECT_EQ(1, tiny.Exponent()); // the smallest normal exponent
  // Stepping down from TINY leaves the normal range.
  EXPECT_FALSE(tiny.NEAREST(false).value.IsNormal());
}

TEST_P(RealValueKind, NotANumber) {
  const int kind{GetParam()};

  RealValue nan{RealValue::NotANumber(kind)};
  EXPECT_EQ(kind, nan.kind());
  EXPECT_TRUE(nan.IsNotANumber());
  EXPECT_FALSE(nan.IsSignalingNaN());
  EXPECT_FALSE(nan.IsFinite());
}

TEST_P(RealValueKind, Exponent) {
  const int kind{GetParam()};

  // Exponent() is the raw, biased exponent field.  The bias is recovered
  // from the (unbiased) Fortran MAXEXPONENT and the raw exponent of
  // Infinity(), which is the maximum representable raw exponent field.
  const int maxRawExponent{RealValue::Infinity(kind).Exponent()};
  const int bias{maxRawExponent - RealValue::MAXEXPONENT(kind)};
  EXPECT_EQ(0, RealValue::Zero(kind).Exponent());
  EXPECT_EQ(bias, (RealValue{kind, 1.0}.Exponent()));
  EXPECT_EQ(bias + 1, (RealValue{kind, 2.0}.Exponent()));
  EXPECT_EQ(maxRawExponent, RealValue::Infinity(kind).Exponent());
  EXPECT_EQ(maxRawExponent, RealValue::NotANumber(kind).Exponent());
}

TEST_P(RealValueKind, EXPONENT) {
  const int kind{GetParam()};

  // The Fortran EXPONENT() intrinsic returns the unbiased exponent, plus one.
  EXPECT_EQ(1, (RealValue{kind, 1.0}.EXPONENT().ToInt64()));
  EXPECT_EQ(2, (RealValue{kind, 2.0}.EXPONENT().ToInt64()));
  EXPECT_EQ(3, (RealValue{kind, 4.0}.EXPONENT().ToInt64()));
  EXPECT_EQ(0, RealValue::Zero(kind).EXPONENT().ToInt64());
  EXPECT_EQ(4, (RealValue{kind, 1.0}.EXPONENT().kind())); // INTEGER(4) result
}

TEST_P(RealValueKind, RRSPACING) {
  const int kind{GetParam()};

  // RRSPACING(1.0) is 2**(DIGITS-1).
  RealValue scaled{RealValue{kind, 1.0}
          .SCALE(IntegerValue{4, RealValue::DIGITS(kind) - 1})
          .value};
  EXPECT_REAL_EQ(scaled, (RealValue{kind, 1.0}.RRSPACING()));
  EXPECT_FALSE((RealValue{kind, -1.0}.RRSPACING().IsNegative()));
  EXPECT_TRUE(RealValue::Infinity(kind).RRSPACING().IsNotANumber());
}

TEST_P(RealValueKind, SPACING) {
  const int kind{GetParam()};

  EXPECT_REAL_EQ(RealValue::EPSILON(kind), (RealValue{kind, 1.0}.SPACING()));
  // The spacing of a zero or subnormal value is defined to be TINY.
  EXPECT_REAL_EQ(RealValue::TINY(kind), RealValue::Zero(kind).SPACING());
  EXPECT_TRUE(RealValue::Infinity(kind).SPACING().IsNotANumber());
}

TEST_P(RealValueKind, SET_EXPONENT) {
  const int kind{GetParam()};

  // SET_EXPONENT(X,I) is FRACTION(X)*2**I.
  EXPECT_REAL_EQ(
      (RealValue{kind, 4.0}), (RealValue{kind, 1.0}.SET_EXPONENT(3)));
  EXPECT_REAL_EQ(
      (RealValue{kind, 1.0}), (RealValue{kind, 8.0}.SET_EXPONENT(1)));
  EXPECT_TRUE(RealValue::Zero(kind).SET_EXPONENT(3).IsZero());
  EXPECT_TRUE(RealValue::Infinity(kind).SET_EXPONENT(3).IsNotANumber());
}

TEST_P(RealValueKind, FRACTION) {
  const int kind{GetParam()};

  // FRACTION() normalizes into [0.5, 1.0).
  EXPECT_REAL_EQ((RealValue{kind, 0.5}), (RealValue{kind, 1.0}.FRACTION()));
  EXPECT_REAL_EQ((RealValue{kind, 0.75}), (RealValue{kind, 3.0}.FRACTION()));
  EXPECT_TRUE(RealValue::Zero(kind).FRACTION().IsZero());
}

TEST_P(RealValueKind, SCALE) {
  const int kind{GetParam()};

  auto scaled{RealValue{kind, 3.0}.SCALE(IntegerValue{4, 4})};
  EXPECT_TRUE(scaled.flags.empty());
  EXPECT_REAL_EQ((RealValue{kind, 48.0}), scaled.value);
  RealValue rescaled{RealValue{kind, 48.0}.SCALE(IntegerValue{4, -4}).value};
  EXPECT_REAL_EQ((RealValue{kind, 3.0}), rescaled);
  // Scaling a zero ignores the factor.
  EXPECT_TRUE(
      RealValue::Zero(kind).SCALE(IntegerValue{4, 1000}).value.IsZero());
}

TEST_P(RealValueKind, FlushSubnormalToZero) {
  const int kind{GetParam()};

  RealValue subnormal{kind, IntegerValue{kind, 1}};
  ASSERT_FALSE(subnormal.IsZero());
  EXPECT_TRUE(subnormal.FlushSubnormalToZero().IsZero());
  // Normal values pass through unchanged.
  EXPECT_REAL_EQ(
      (RealValue{kind, 3.0}), (RealValue{kind, 3.0}.FlushSubnormalToZero()));
  EXPECT_REAL_EQ(
      RealValue::TINY(kind), RealValue::TINY(kind).FlushSubnormalToZero());
}

//===----------------------------------------------------------------------===//
// Conversions
//===----------------------------------------------------------------------===//

TEST_P(RealValueKind, FromInteger) {
  const int kind{GetParam()};

  auto exact{RealValue::FromInteger(kind, IntegerValue{8, 3})};
  EXPECT_TRUE(exact.flags.empty());
  EXPECT_EQ(kind, exact.value.kind());
  EXPECT_EQ(Relation::Equal, exact.value.Compare(RealValue{kind, 3.0}));
  EXPECT_TRUE(
      RealValue::FromInteger(kind, IntegerValue::Zero(8)).value.IsZero());

  auto negative{RealValue::FromInteger(kind, IntegerValue{8, -3})};
  EXPECT_TRUE(negative.value.IsNegative());

  // The same bit pattern read as unsigned is a large positive number.
  auto asUnsigned{
      RealValue::FromInteger(kind, IntegerValue{8, -1}, /*isUnsigned=*/true)};
  EXPECT_FALSE(asUnsigned.value.IsNegative());
  EXPECT_FALSE(asUnsigned.value.IsZero());
}

TEST_P(RealValueKind, ToWholeNumber) {
  const int kind{GetParam()};

  // 3.5 is representable in every supported format.
  RealValue x{kind, 3.5};
  EXPECT_REAL_EQ((RealValue{kind, 3.0}), x.ToWholeNumber().value);
  EXPECT_REAL_EQ(
      (RealValue{kind, 4.0}), x.ToWholeNumber(RoundingMode::TiesToEven).value);
  EXPECT_REAL_EQ(
      (RealValue{kind, 4.0}), x.ToWholeNumber(RoundingMode::Up).value);
  EXPECT_REAL_EQ(
      (RealValue{kind, 3.0}), x.ToWholeNumber(RoundingMode::Down).value);

  RealValue negative{x.Negate()};
  EXPECT_REAL_EQ((RealValue{kind, -3.0}), negative.ToWholeNumber().value);
  EXPECT_REAL_EQ((RealValue{kind, -4.0}),
      negative.ToWholeNumber(RoundingMode::Down).value);
  EXPECT_REAL_EQ(
      (RealValue{kind, -3.0}), negative.ToWholeNumber(RoundingMode::Up).value);
  // Whole numbers, infinities and NaNs.
  EXPECT_REAL_EQ(
      (RealValue{kind, 3.0}), (RealValue{kind, 3.0}.ToWholeNumber().value));
  EXPECT_TRUE(
      RealValue::Infinity(kind).ToWholeNumber().flags.test(RealFlag::Overflow));
  EXPECT_TRUE(RealValue::NotANumber(kind).ToWholeNumber().flags.test(
      RealFlag::InvalidArgument));
}

TEST_P(RealValueKind, ToInteger) {
  const int kind{GetParam()};

  auto exact{RealValue{kind, 42.0}.ToInteger()};
  EXPECT_TRUE(exact.flags.empty());
  EXPECT_EQ(42, exact.value.ToInt64());
  EXPECT_EQ(8, exact.value.kind()); // an INTEGER(8) by default
  EXPECT_EQ(4,
      (RealValue{kind, 42.0}.ToInteger(RoundingMode::ToZero, 32).value.kind()));
  EXPECT_EQ(-42, (RealValue{kind, -42.0}.ToInteger().value.ToInt64()));

  // Rounding modes.
  RealValue x{kind, 3.5};
  EXPECT_EQ(3, x.ToInteger(RoundingMode::ToZero).value.ToInt64());
  EXPECT_EQ(4, x.ToInteger(RoundingMode::TiesToEven).value.ToInt64());
  EXPECT_EQ(4, x.ToInteger(RoundingMode::Up).value.ToInt64());
  EXPECT_EQ(3, x.ToInteger(RoundingMode::Down).value.ToInt64());

  // A NaN is invalid and yields HUGE.
  auto nan{RealValue::NotANumber(kind).ToInteger()};
  EXPECT_TRUE(nan.flags.test(RealFlag::InvalidArgument));
  EXPECT_TRUE(nan.value == IntegerValue::HUGE(8));
  // An infinity overflows.
  EXPECT_TRUE(
      RealValue::Infinity(kind).ToInteger().flags.test(RealFlag::Overflow));
  // So does a value too large for the target integer.
  EXPECT_TRUE(RealValue::HUGE(kind)
          .ToInteger(RoundingMode::ToZero, 8)
          .flags.test(RealFlag::Overflow));
}

TEST_P(RealValueKind, Convert) {
  const int kind{GetParam()};

  // Widening to REAL(16) and narrowing back is lossless.
  auto widened{RealValue::Convert(16, RealValue{kind, 3.0})};
  EXPECT_EQ(16, widened.value.kind());
  auto restored{RealValue::Convert(kind, widened.value)};
  EXPECT_REAL_EQ((RealValue{kind, 3.0}), restored.value);
  // Converting to the same kind is the identity.
  EXPECT_REAL_EQ((RealValue{kind, 3.0}),
      (RealValue::Convert(kind, RealValue{kind, 3.0}).value));

  // A NaN is invalid but stays a NaN.
  auto nan{RealValue::Convert(kind, RealValue::NotANumber(16))};
  EXPECT_TRUE(nan.flags.test(RealFlag::InvalidArgument));
  EXPECT_TRUE(nan.value.IsNotANumber());
  // Overflow when the source magnitude exceeds the destination's range.
  if (kind != 16) {
    auto overflowed{RealValue::Convert(kind, RealValue::HUGE(16))};
    EXPECT_TRUE(overflowed.flags.test(RealFlag::Overflow));
    EXPECT_TRUE(overflowed.value.IsInfinite());
  }
}

//===----------------------------------------------------------------------===//
// Raw bits, formatting and parsing
//===----------------------------------------------------------------------===//

TEST_P(RealValueKind, RawBits) {
  const int kind{GetParam()};

  EXPECT_TRUE(RealValue::Zero(kind).RawBits().IsZero());

  // REAL(10) stores 128 bits, but only 80 of them are significant.
  const int significantBits{kind == 10 ? 80 : RealValue::bits(kind)};
  RealValue allOnes{kind, IntegerValue::MASKR(kind, significantBits)};
  EXPECT_EQ(significantBits, allOnes.RawBits().POPCNT());
  EXPECT_EQ(1, RealValue::NegativeZero(kind).RawBits().POPCNT());
  EXPECT_EQ(0, RealValue::NegativeZero(kind).RawBits().LEADZ());

  // The bit pattern round-trips through the (kind, Word) constructor.
  RealValue x{kind, 3.0};
  EXPECT_REAL_EQ(x, (RealValue{kind, x.RawBits()}));
}

TEST_P(RealValueKind, RawBytesRoundTrip) {
  const int kind{GetParam()};
  RealValue original{kind, -3.0};
  char buffer[16]{};
  ASSERT_EQ(RealValue::bytesStored(kind), original.bytesStored());

  bool changed1{false};
  original.StoreRawBytes(buffer, original.bytesStored(), &changed1);
  EXPECT_TRUE(changed1);

  RealValue restored{
      RealValue::FromRawBytes(kind, buffer, original.bytesStored())};
  EXPECT_EQ(kind, restored.kind());
  EXPECT_REAL_EQ(original, restored);

  bool changed2{false};
  original.StoreRawBytes(buffer, original.bytesStored(), &changed2);
  EXPECT_FALSE(changed2);
}

// Ported from the legacy non-GTest test flang/unittests/Evaluate/real.cpp.
TEST(RealValue, DumpHexadecimal) {
  struct {
    std::uint64_t raw;
    const char *expected;
  } table[]{
      {0x7f876543, "NaN0x7f876543"},
      {0x7f800000, "Inf"},
      {0xff800000, "-Inf"},
      {0x00000000, "0.0"},
      {0x80000000, "-0.0"},
      {0x3f800000, "0x1.0p0"},
      {0xbf800000, "-0x1.0p0"},
      {0x40000000, "0x1.0p1"},
      {0x3f000000, "0x1.0p-1"},
      {0x7f7fffff, "0x1.fffffep127"},
      {0x00800000, "0x1.0p-126"},
      {0x00400000, "0x0.8p-126"},
      {0x00000001, "0x0.000002p-126"},
  };
  for (auto &e : table) {
    EXPECT_EQ(
        e.expected, (RealValue{4, IntegerValue{4, e.raw}}.DumpHexadecimal()))
        << "raw=" << e.raw;
  }
}

TEST_P(RealValueKind, AsFortran) {
  const int kind{GetParam()};

  // NaNs and infinities are emitted as parenthesized expressions.
  std::string nan{AsFortranString(RealValue::NotANumber(kind), kind, false)};
  EXPECT_EQ("(0._" + std::to_string(kind) + "/0.)", nan);
  std::string inf{AsFortranString(RealValue::Infinity(kind), kind, false)};
  EXPECT_EQ("(1._" + std::to_string(kind) + "/0.)", inf);

  std::string negInf{
      AsFortranString(RealValue::Infinity(kind, true), kind, false)};
  EXPECT_EQ("(-1._" + std::to_string(kind) + "/0.)", negInf);

  // A finite value reads back as itself.
  RealValue x{kind, 0.375};
  std::string decimal{AsFortranString(x, kind, false)};
  const char *p{decimal.c_str()};
  if (*p == '(') {
    ++p;
  }

  auto readBack{RealValue::Read(kind, p)};
  EXPECT_REAL_EQ(x, readBack.value);
  EXPECT_EQ('_', *p) << decimal;
  // The minimal form also reads back as itself.
  std::string minimal{AsFortranString(x, kind, true)};
  p = minimal.c_str();
  if (*p == '(') {
    ++p;
  }
  EXPECT_REAL_EQ(x, RealValue::Read(kind, p).value);
}

TEST_P(RealValueKind, Read) {
  const int kind{GetParam()};
  const char *text{"1.0rest"};
  const char *p{text};
  auto one{RealValue::Read(kind, p)};
  EXPECT_EQ(kind, one.value.kind());
  EXPECT_REAL_EQ((RealValue{kind, 1.0}), one.value);
  EXPECT_STREQ("rest", p);

  const char *negative{"-2.5"};
  p = negative;
  auto minusTwoAndAHalf{RealValue::Read(kind, p)};
  EXPECT_REAL_EQ((RealValue{kind, -2.5}), minusTwoAndAHalf.value);

  // 0.1 is inexact in every binary format.
  const char *tenth{"0.1"};
  p = tenth;
  EXPECT_TRUE(RealValue::Read(kind, p).flags.test(RealFlag::Inexact));
}

TEST_P(RealValueKind, RoundingModes) {
  const int kind{GetParam()};

  // 1 + EPSILON/2 is exactly halfway between 1 and the next value up, so each
  // rounding mode picks a different result.
  RealValue one{kind, 1.0};
  RealValue half{RealValue::EPSILON(kind).Divide(RealValue{kind, 2.0}).value};
  RealValue up{one.Add(RealValue::EPSILON(kind)).value};
  EXPECT_REAL_EQ(
      one, one.Add(half, Rounding{RoundingMode::TiesToEven}).value); // to even
  EXPECT_REAL_EQ(one, one.Add(half, Rounding{RoundingMode::ToZero}).value);
  EXPECT_REAL_EQ(one, one.Add(half, Rounding{RoundingMode::Down}).value);
  EXPECT_REAL_EQ(up, one.Add(half, Rounding{RoundingMode::Up}).value);
  EXPECT_REAL_EQ(
      up, one.Add(half, Rounding{RoundingMode::TiesAwayFromZero}).value);
}

TEST_P(RealValueKind, Print) {
  const int kind{GetParam()};
  const int pos{KindPos(kind)};

  llvm::SmallString<128> buf;
  llvm::raw_svector_ostream os{buf};
  RealValue v{kind, 42.0};
  v.print(os);

  const char *results[]{
      "4.2e1_2", "4.2e1_3", "4.2e1_4", "4.2e1_8", "4.2e1_10", "4.2e1_16"};
  EXPECT_EQ(results[pos], os.str());
}

//===----------------------------------------------------------------------===//
// Ported coverage from flang/unittests/Evaluate/real.cpp
//===----------------------------------------------------------------------===//

// Mirrors basicTests() from the legacy test: converts every power of two that
// fits in an INTEGER(8) and converts it back.
TEST_P(RealValueKind, FromIntegerPowersOfTwo) {
  const int kind{GetParam()};
  const int bias{
      RealValue::Infinity(kind).Exponent() - RealValue::MAXEXPONENT(kind)};
  for (int j{0}; j < 63; ++j) {
    SCOPED_TRACE(testing::Message() << "kind=" << kind << " 2**" << j);
    const std::uint64_t x{std::uint64_t{1} << j};
    IntegerValue ix{8, x};
    ASSERT_FALSE(ix.IsNegative());
    ASSERT_EQ(x, ix.ToUInt64());

    auto vr{RealValue::FromInteger(kind, ix)};
    EXPECT_FALSE(vr.value.IsNegative());
    EXPECT_FALSE(vr.value.IsNotANumber());
    EXPECT_FALSE(vr.value.IsZero());
    auto back{vr.value.ToInteger()};
    if (j > bias) {
      EXPECT_TRUE(vr.flags.test(RealFlag::Overflow));
      EXPECT_TRUE(vr.value.IsInfinite());
      EXPECT_TRUE(back.flags.test(RealFlag::Overflow));
      EXPECT_EQ(0x7fffffffffffffffu, back.value.ToUInt64());
    } else {
      EXPECT_TRUE(vr.flags.empty());
      EXPECT_FALSE(vr.value.IsInfinite());
      EXPECT_TRUE(back.flags.empty());
      EXPECT_EQ(x, back.value.ToUInt64());
      // A power of two is a whole number already.
      EXPECT_EQ(
          Relation::Equal, vr.value.ToWholeNumber().value.Compare(vr.value));
      // Emitting and re-reading the value is lossless.
      std::string decimal{AsFortranString(vr.value, kind, false)};
      const char *p{decimal.c_str()};
      auto check{RealValue::Read(kind, p)};
      EXPECT_EQ(Relation::Equal, vr.value.Compare(check.value)) << decimal;
      EXPECT_EQ(x, check.value.ToInteger().value.ToUInt64()) << decimal;
    }

    IntegerValue negIx{ix.Negate().value};
    ASSERT_TRUE(negIx.IsNegative());
    auto negVr{RealValue::FromInteger(kind, negIx)};
    EXPECT_TRUE(negVr.value.IsNegative());
    EXPECT_FALSE(negVr.value.IsNotANumber());
    EXPECT_FALSE(negVr.value.IsZero());
    auto negBack{negVr.value.ToInteger()};
    if (j > bias) {
      EXPECT_TRUE(negVr.flags.test(RealFlag::Overflow));
      EXPECT_TRUE(negVr.value.IsInfinite());
      EXPECT_TRUE(negBack.flags.test(RealFlag::Overflow));
      EXPECT_EQ(0x8000000000000000u, negBack.value.ToUInt64());
    } else {
      EXPECT_TRUE(negVr.flags.empty());
      EXPECT_FALSE(negVr.value.IsInfinite());
      EXPECT_TRUE(negBack.flags.empty());
      EXPECT_EQ(negIx.ToInt64(), negBack.value.ToInt64());
    }
    EXPECT_EQ(Relation::Equal,
        negVr.value.ToWholeNumber().value.Compare(negVr.value));
  }
}

// Mirrors subsetTests() from the real.cpp legacy test, comparing against the
// host's hardware arithmetic in the default (round-to-nearest) mode.
TYPED_TEST(RealValueHostTypedKind, CompareUnaryWithHost) {
  using HostT = typename TypeParam::HostT;
  using UnsignedT = typename TypeParam::UnsignedT;
  constexpr int kind{TypeParam::kind};

  union {
    UnsignedT ui;
    HostT f;
  } u;
  constexpr UnsignedT operands{4096};
  for (UnsignedT j{0}; j < operands; ++j) {
    const UnsignedT raw{SpreadBits(j)};
    u.ui = raw;
    const HostT f{u.f};
    RealValue x{kind, IntegerValue{kind, std::uint64_t{raw}}};
    SCOPED_TRACE(testing::Message()
        << "kind=" << kind << " raw=0x" << x.RawBits().Hexadecimal());

    ASSERT_EQ(std::uint64_t{raw}, x.RawBits().ToUInt64());
    EXPECT_EQ(std::isnan(f), x.IsNotANumber());
    EXPECT_EQ(std::isinf(f), x.IsInfinite());
    EXPECT_EQ(std::isfinite(f), x.IsFinite());
    EXPECT_EQ(f == 0, x.IsZero());
    EXPECT_EQ(std::signbit(f) && !std::isnan(f), x.IsNegative());
    EXPECT_EQ(
        std::isfinite(f) && std::fpclassify(f) != FP_SUBNORMAL, x.IsNormal());

    ExpectSameAsHost<HostT, UnsignedT>(x.ToWholeNumber().value, std::trunc(f));
    ExpectSameAsHost<HostT, UnsignedT>(x.SQRT().value, std::sqrt(f));
    ExpectSameAsHost<HostT, UnsignedT>(x.ABS(), std::fabs(f));
    if (!std::isnan(f)) {
      ExpectSameAsHost<HostT, UnsignedT>(x.Negate(), -f);
    }

    // Every value is emitted as a Fortran constant that reads back exactly.
    const std::string kindSuffix{std::to_string(kind)};
    std::string text{AsFortranString(x, kind, false)};
    if (std::isnan(f)) {
      EXPECT_EQ("(0._" + kindSuffix + "/0.)", text);
    } else if (std::isinf(f)) {
      EXPECT_EQ(
          (std::signbit(f) ? "(-1._" : "(1._") + kindSuffix + "/0.)", text);
    } else {
      const char *p{text.c_str()};
      if (*p == '(') {
        ++p;
      }
      auto readBack{RealValue::Read(kind, p)};
      EXPECT_EQ(std::uint64_t{raw}, readBack.value.RawBits().ToUInt64())
          << text;
      EXPECT_EQ('_', *p) << text;
    }
  }
}

TYPED_TEST(RealValueHostTypedKind, CompareDyadicWithHost) {
  using HostT = typename TypeParam::HostT;
  using UnsignedT = typename TypeParam::UnsignedT;
  constexpr int kind{TypeParam::kind};

  union {
    UnsignedT ui;
    HostT f;
  } u;
  constexpr UnsignedT operands{128};
  for (UnsignedT j{0}; j < operands; ++j) {
    const UnsignedT rj{SpreadBits(j)};
    u.ui = rj;
    const HostT fj{u.f};
    RealValue x{kind, IntegerValue{kind, std::uint64_t{rj}}};
    for (UnsignedT k{0}; k < operands; ++k) {
      const UnsignedT rk{SpreadBits(k)};
      u.ui = rk;
      const HostT fk{u.f};
      RealValue y{kind, IntegerValue{kind, std::uint64_t{rk}}};
      SCOPED_TRACE(testing::Message()
          << "kind=" << kind << " x=0x" << x.RawBits().Hexadecimal() << " y=0x"
          << y.RawBits().Hexadecimal());
      ExpectSameAsHost<HostT, UnsignedT>(x.Add(y).value, fj + fk);
      ExpectSameAsHost<HostT, UnsignedT>(x.Subtract(y).value, fj - fk);
      ExpectSameAsHost<HostT, UnsignedT>(x.Multiply(y).value, fj * fk);
      ExpectSameAsHost<HostT, UnsignedT>(x.Divide(y).value, fj / fk);
    }
  }
}

} // namespace
