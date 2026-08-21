//===-- flang/unittests/Evaluate/ComplexValueTest.cpp ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "flang/Common/type-kinds.h"
#include "flang/Evaluate/complex-value.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

using namespace Fortran::common;
using namespace Fortran::evaluate;
using namespace Fortran::evaluate::value;

namespace {

class ComplexValueKind : public testing::TestWithParam<KindsEnum> {};
INSTANTIATE_TEST_SUITE_P(ComplexValueKind, ComplexValueKind,
    testing::ValuesIn(RealKinds),
    [](const testing::TestParamInfo<KindsEnum> &info) {
      return "COMPLEX(" + std::to_string(static_cast<int>(info.param)) + ")";
    });

RealValue Real(KindsEnum kind, std::int64_t n) {
  return RealValue::FromInteger(kind, IntegerValue{KindsEnum::Kind8, n}).value;
}

ComplexValue Complex(KindsEnum kind, std::int64_t re, std::int64_t im) {
  return ComplexValue{Real(kind, re), Real(kind, im)};
}

testing::AssertionResult ComplexValuesEqual(const char *lhsExpr,
    const char *rhsExpr, const ComplexValue &lhs, const ComplexValue &rhs) {
  if (lhs == rhs) {
    return testing::AssertionSuccess();
  }
  return testing::AssertionFailure()
      << lhsExpr << " (" << lhs.DumpHexadecimal() << ") != " << rhsExpr << " ("
      << rhs.DumpHexadecimal() << ")";
}

#define EXPECT_COMPLEX_EQ(lhs, rhs) \
  EXPECT_PRED_FORMAT2(ComplexValuesEqual, lhs, rhs)

std::string AsFortranString(const ComplexValue &z, KindsEnum kind) {
  std::string s;
  llvm::raw_string_ostream os{s};
  z.AsFortran(os, static_cast<int>(kind));
  return s;
}

constexpr int KindPos(KindsEnum kind) {
  for (std::size_t i{0}; i < std::size(RealKinds); ++i) {
    if (RealKinds[i] == kind) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

//===----------------------------------------------------------------------===//
// Construction and kind inquiries
//===----------------------------------------------------------------------===//

TEST(ComplexValue, DefaultConstructionIsMonostate) {
  ComplexValue z;
  EXPECT_TRUE(z.IsMonostate());
  EXPECT_TRUE(z.IsZero());
  EXPECT_FALSE(z.IsInfinite());
  EXPECT_FALSE(z.IsNotANumber());
  EXPECT_FALSE(z.IsSignalingNaN());
}

TEST_P(ComplexValueKind, ConstructFromParts) {
  const KindsEnum kind{GetParam()};
  ComplexValue z{Real(kind, 1), Real(kind, 2)};
  EXPECT_FALSE(z.IsMonostate());
  EXPECT_EQ(kind, z.kind());
  EXPECT_TRUE(z.REAL() == Real(kind, 1));
  EXPECT_TRUE(z.AIMAG() == Real(kind, 2));
}

TEST_P(ComplexValueKind, ConstructFromRealPartOnly) {
  const KindsEnum kind{GetParam()};
  ComplexValue z{Real(kind, 3)};
  EXPECT_EQ(kind, z.kind());
  EXPECT_TRUE(z.REAL() == Real(kind, 3));
  EXPECT_TRUE(z.AIMAG().IsZero());
  // The kind-checking form agrees.
  EXPECT_COMPLEX_EQ(z, ComplexValue(kind, Real(kind, 3)));
}

TEST_P(ComplexValueKind, ImaginaryPartIsConvertedToTheRealPartsKind) {
  const KindsEnum kind{GetParam()};
  // The imaginary operand is converted to the kind of the real operand.
  ComplexValue z{Real(kind, 1), Real(KindsEnum::Kind8, 2)};
  EXPECT_EQ(kind, z.kind());
  EXPECT_TRUE(z.AIMAG() == Real(kind, 2));
}

TEST(ComplexValue, CopyAndMove) {
  ComplexValue z{Complex(KindsEnum::Kind4, 1, 2)};
  ComplexValue copyConstructed{z};
  EXPECT_COMPLEX_EQ(z, copyConstructed);
  ComplexValue copyAssigned;
  copyAssigned = z;
  EXPECT_COMPLEX_EQ(z, copyAssigned);
  ComplexValue moveConstructed{std::move(copyConstructed)};
  EXPECT_COMPLEX_EQ(z, moveConstructed);
  ComplexValue moveAssigned;
  moveAssigned = std::move(copyAssigned);
  EXPECT_COMPLEX_EQ(z, moveAssigned);
}

TEST(ComplexValue, KindCheckingConstructors) {
  ComplexValue z{Complex(KindsEnum::Kind4, 1, 2)};
  EXPECT_EQ(KindsEnum::Kind4, ComplexValue(KindsEnum::Kind4, z).kind());
  EXPECT_COMPLEX_EQ(z, ComplexValue(KindsEnum::Kind4, z));
  ComplexValue y{Complex(KindsEnum::Kind8, 1, 2)};
  ComplexValue moved{KindsEnum::Kind8, std::move(y)};
  EXPECT_EQ(KindsEnum::Kind8, moved.kind());
}

TEST_P(ComplexValueKind, Zero) {
  const KindsEnum kind{GetParam()};
  ComplexValue zero{ComplexValue::Zero(kind)};
  EXPECT_FALSE(zero.IsMonostate());
  EXPECT_EQ(kind, zero.kind());
  EXPECT_TRUE(zero.IsZero());
  EXPECT_FALSE(zero.REAL().IsNegative());
  EXPECT_FALSE(zero.AIMAG().IsNegative());
}

TEST(ComplexValue, BytesStored) {
  EXPECT_EQ(4u, ComplexValue::bytesStored(KindsEnum::Kind2));
  EXPECT_EQ(4u, ComplexValue::bytesStored(KindsEnum::Kind3));
  EXPECT_EQ(8u, ComplexValue::bytesStored(KindsEnum::Kind4));
  EXPECT_EQ(16u, ComplexValue::bytesStored(KindsEnum::Kind8));
  EXPECT_EQ(32u, ComplexValue::bytesStored(KindsEnum::Kind10));
  EXPECT_EQ(32u, ComplexValue::bytesStored(KindsEnum::Kind16));
  EXPECT_EQ(8u, Complex(KindsEnum::Kind4, 1, 2).bytesStored());
}

//===----------------------------------------------------------------------===//
// Component access and sign manipulation
//===----------------------------------------------------------------------===//

TEST_P(ComplexValueKind, REAL) {
  const KindsEnum kind{GetParam()};
  EXPECT_TRUE(Complex(kind, 1, 2).REAL() == Real(kind, 1));
  EXPECT_EQ(kind, Complex(kind, 1, 2).REAL().kind());
}

TEST_P(ComplexValueKind, AIMAG) {
  const KindsEnum kind{GetParam()};
  EXPECT_TRUE(Complex(kind, 1, 2).AIMAG() == Real(kind, 2));
  EXPECT_EQ(kind, Complex(kind, 1, 2).AIMAG().kind());
}

TEST_P(ComplexValueKind, CONJG) {
  const KindsEnum kind{GetParam()};
  EXPECT_COMPLEX_EQ(Complex(kind, 1, -2), Complex(kind, 1, 2).CONJG());
  EXPECT_COMPLEX_EQ(Complex(kind, 1, 2), Complex(kind, 1, 2).CONJG().CONJG());
}

TEST_P(ComplexValueKind, Negate) {
  const KindsEnum kind{GetParam()};
  EXPECT_COMPLEX_EQ(Complex(kind, -1, -2), Complex(kind, 1, 2).Negate());
  // Negating a zero flips both sign bits.
  ComplexValue negZero{ComplexValue::Zero(kind).Negate()};
  EXPECT_TRUE(negZero.IsZero());
  EXPECT_TRUE(negZero.REAL().IsNegative());
  EXPECT_TRUE(negZero.AIMAG().IsNegative());
}

//===----------------------------------------------------------------------===//
// Comparison and classification
//===----------------------------------------------------------------------===//

TEST_P(ComplexValueKind, Equals) {
  const KindsEnum kind{GetParam()};
  // Equals() compares numerically, so +0.0 and -0.0 are equal ...
  EXPECT_TRUE(
      ComplexValue::Zero(kind).Equals(ComplexValue::Zero(kind).Negate()));
  EXPECT_TRUE(Complex(kind, 1, 2).Equals(Complex(kind, 1, 2)));
  EXPECT_FALSE(Complex(kind, 1, 2).Equals(Complex(kind, 1, 3)));
  // ... and a NaN is equal to nothing, not even itself.
  EXPECT_FALSE(
      ComplexValue::NotANumber(kind).Equals(ComplexValue::NotANumber(kind)));
}

TEST_P(ComplexValueKind, EqualityOperators) {
  const KindsEnum kind{GetParam()};
  // The operators compare bit patterns, so -0.0 differs from +0.0 ...
  EXPECT_FALSE(ComplexValue::Zero(kind) == ComplexValue::Zero(kind).Negate());
  EXPECT_TRUE(ComplexValue::Zero(kind) != ComplexValue::Zero(kind).Negate());
  // ... and a NaN equals itself.
  EXPECT_TRUE(ComplexValue::NotANumber(kind) == ComplexValue::NotANumber(kind));
  EXPECT_TRUE(Complex(kind, 1, 2) == Complex(kind, 1, 2));
  EXPECT_TRUE(Complex(kind, 1, 2) != Complex(kind, 2, 1));
}

TEST_P(ComplexValueKind, IsZero) {
  const KindsEnum kind{GetParam()};
  EXPECT_TRUE(ComplexValue::Zero(kind).IsZero());
  EXPECT_FALSE(Complex(kind, 1, 0).IsZero());
  EXPECT_FALSE(Complex(kind, 0, 1).IsZero());
}

TEST_P(ComplexValueKind, IsInfinite) {
  const KindsEnum kind{GetParam()};
  RealValue inf{Real(kind, 1).Divide(RealValue::Zero(kind)).value};
  ASSERT_TRUE(inf.IsInfinite());
  EXPECT_FALSE(ComplexValue::Zero(kind).IsInfinite());
  // Either part being infinite suffices.
  EXPECT_TRUE(ComplexValue(inf, Real(kind, 1)).IsInfinite());
  EXPECT_TRUE(ComplexValue(Real(kind, 1), inf).IsInfinite());
}

TEST_P(ComplexValueKind, IsNotANumber) {
  const KindsEnum kind{GetParam()};
  RealValue nan{RealValue::NotANumber(kind)};
  EXPECT_FALSE(ComplexValue::Zero(kind).IsNotANumber());
  EXPECT_TRUE(ComplexValue::NotANumber(kind).IsNotANumber());
  // Either part being a NaN suffices.
  EXPECT_TRUE(ComplexValue(nan, Real(kind, 1)).IsNotANumber());
  EXPECT_TRUE(ComplexValue(Real(kind, 1), nan).IsNotANumber());
}

TEST_P(ComplexValueKind, IsSignalingNaN) {
  const KindsEnum kind{GetParam()};
  EXPECT_FALSE(ComplexValue::Zero(kind).IsSignalingNaN());
  // NotANumber() produces quiet NaNs.
  EXPECT_FALSE(ComplexValue::NotANumber(kind).IsSignalingNaN());
}

TEST_P(ComplexValueKind, NotANumber) {
  const KindsEnum kind{GetParam()};
  ComplexValue nan{ComplexValue::NotANumber(kind)};
  EXPECT_EQ(kind, nan.kind());
  EXPECT_TRUE(nan.REAL().IsNotANumber());
  EXPECT_TRUE(nan.AIMAG().IsNotANumber());
}

//===----------------------------------------------------------------------===//
// Arithmetic
//===----------------------------------------------------------------------===//

TEST_P(ComplexValueKind, FromInteger) {
  const KindsEnum kind{GetParam()};
  auto z{ComplexValue::FromInteger(kind, IntegerValue{KindsEnum::Kind8, 3})};
  EXPECT_TRUE(z.flags.empty());
  EXPECT_EQ(kind, z.value.kind());
  EXPECT_COMPLEX_EQ(Complex(kind, 3, 0), z.value);
  auto negative{
      ComplexValue::FromInteger(kind, IntegerValue{KindsEnum::Kind8, -3})};
  EXPECT_COMPLEX_EQ(Complex(kind, -3, 0), negative.value);
  // Reading the same bits as unsigned gives a large positive real part.
  auto asUnsigned{ComplexValue::FromInteger(
      kind, IntegerValue{KindsEnum::Kind8, -1}, /*isUnsigned=*/true)};
  EXPECT_FALSE(asUnsigned.value.REAL().IsNegative());
  EXPECT_TRUE(asUnsigned.value.AIMAG().IsZero());
}

TEST_P(ComplexValueKind, Add) {
  const KindsEnum kind{GetParam()};
  auto sum{Complex(kind, 1, 2).Add(Complex(kind, 3, 4))};
  EXPECT_TRUE(sum.flags.empty());
  EXPECT_COMPLEX_EQ(Complex(kind, 4, 6), sum.value);
  // Flags from either part are accumulated.
  auto overflowed{ComplexValue(RealValue::HUGE(kind))
          .Add(ComplexValue(RealValue::HUGE(kind)))};
  EXPECT_TRUE(overflowed.flags.test(RealFlag::Overflow));
  EXPECT_TRUE(overflowed.value.IsInfinite());
}

TEST_P(ComplexValueKind, Subtract) {
  const KindsEnum kind{GetParam()};
  auto diff{Complex(kind, 1, 2).Subtract(Complex(kind, 3, 4))};
  EXPECT_TRUE(diff.flags.empty());
  EXPECT_COMPLEX_EQ(Complex(kind, -2, -2), diff.value);
}

TEST_P(ComplexValueKind, Multiply) {
  const KindsEnum kind{GetParam()};
  // (1+2i)*(3+4i) = (3-8) + (4+6)i
  auto product{Complex(kind, 1, 2).Multiply(Complex(kind, 3, 4))};
  EXPECT_TRUE(product.flags.empty());
  EXPECT_COMPLEX_EQ(Complex(kind, -5, 10), product.value);
  // Multiplying by i rotates by a quarter turn.
  EXPECT_COMPLEX_EQ(Complex(kind, -2, 1),
      Complex(kind, 1, 2).Multiply(Complex(kind, 0, 1)).value);
}

TEST_P(ComplexValueKind, Divide) {
  const KindsEnum kind{GetParam()};
  // (-5+10i)/(3+4i) = 1+2i
  auto quotient{Complex(kind, -5, 10).Divide(Complex(kind, 3, 4))};
  EXPECT_COMPLEX_EQ(Complex(kind, 1, 2), quotient.value);
  // Dividing by a real number divides both parts.
  EXPECT_COMPLEX_EQ(Complex(kind, 1, 2),
      Complex(kind, 4, 8).Divide(Complex(kind, 4, 0)).value);
  // Dividing by a purely imaginary number.
  EXPECT_COMPLEX_EQ(Complex(kind, 2, 0),
      Complex(kind, 0, 4).Divide(Complex(kind, 0, 2)).value);
  // Dividing by zero reaches (0/0) in the numerator, hence a NaN rather than
  // an infinity.
  auto byZero{Complex(kind, 1, 0).Divide(ComplexValue::Zero(kind))};
  EXPECT_TRUE(byZero.flags.test(RealFlag::InvalidArgument));
  EXPECT_TRUE(byZero.value.IsNotANumber());
}

TEST_P(ComplexValueKind, ABS) {
  const KindsEnum kind{GetParam()};
  auto abs{Complex(kind, 3, 4).ABS()};
  EXPECT_TRUE(abs.value == Real(kind, 5));
  EXPECT_EQ(kind, abs.value.kind());
  EXPECT_TRUE(Complex(kind, -3, -4).ABS().value == Real(kind, 5));
  EXPECT_TRUE(ComplexValue::Zero(kind).ABS().value.IsZero());
}

TEST_P(ComplexValueKind, KahanSummation) {
  const KindsEnum kind{GetParam()};
  ComplexValue correction{ComplexValue::Zero(kind)};
  auto sum{Complex(kind, 1, 2).KahanSummation(Complex(kind, 3, 4), correction)};
  EXPECT_COMPLEX_EQ(Complex(kind, 4, 6), sum.value);
  EXPECT_TRUE(correction.IsZero());
  // A contribution too small to appear in the sum survives in the correction.
  RealValue tooSmall{RealValue::EPSILON(kind).Divide(Real(kind, 4)).value};
  correction = ComplexValue::Zero(kind);
  auto lossy{Complex(kind, 1, 1)
          .KahanSummation(ComplexValue{tooSmall, tooSmall}, correction)};
  EXPECT_COMPLEX_EQ(Complex(kind, 1, 1), lossy.value);
  EXPECT_FALSE(correction.IsZero());
}

TEST_P(ComplexValueKind, FlushSubnormalToZero) {
  const KindsEnum kind{GetParam()};
  RealValue subnormal{RealValue{kind, IntegerValue{kind, 1}}};
  ASSERT_FALSE(subnormal.IsZero());
  ComplexValue z{subnormal, subnormal};
  EXPECT_FALSE(z.IsZero());
  EXPECT_TRUE(z.FlushSubnormalToZero().IsZero());
  // Normal values pass through unchanged.
  EXPECT_COMPLEX_EQ(
      Complex(kind, 1, 2), Complex(kind, 1, 2).FlushSubnormalToZero());
}

//===----------------------------------------------------------------------===//
// Formatting and raw storage
//===----------------------------------------------------------------------===//

TEST(ComplexValue, DumpHexadecimal) {
  EXPECT_EQ(
      "(0.0,0.0)", ComplexValue::Zero(KindsEnum::Kind4).DumpHexadecimal());
  EXPECT_EQ(
      "(0x1.0p0,-0x1.0p1)", Complex(KindsEnum::Kind4, 1, -2).DumpHexadecimal());
}

TEST_P(ComplexValueKind, AsFortran) {
  const KindsEnum kind{GetParam()};
  std::string s{AsFortranString(Complex(kind, 1, 2), kind)};
  // The components are emitted as a parenthesized, comma-separated pair.
  EXPECT_EQ('(', s.front());
  EXPECT_EQ(')', s.back());
  EXPECT_NE(std::string::npos, s.find(','));
}

TEST_P(ComplexValueKind, RawBytesRoundTrip) {
  const KindsEnum kind{GetParam()};
  ComplexValue original{Complex(kind, 1, -2)};
  char buffer[32]{};
  ASSERT_EQ(ComplexValue::bytesStored(kind), original.bytesStored());
  bool changed{false};
  original.StoreRawBytes(buffer, original.bytesStored(), &changed);
  EXPECT_TRUE(changed);
  ComplexValue restored{
      ComplexValue::FromRawBytes(kind, buffer, original.bytesStored())};
  EXPECT_EQ(kind, restored.kind());
  EXPECT_COMPLEX_EQ(original, restored);
  changed = false;
  original.StoreRawBytes(buffer, original.bytesStored(), &changed);
  EXPECT_FALSE(changed);
}

TEST_P(ComplexValueKind, Print) {
  const KindsEnum kind{GetParam()};
  const int pos{KindPos(kind)};

  llvm::SmallString<128> buf;
  llvm::raw_svector_ostream os{buf};
  ComplexValue v{RealValue::FromInteger(kind, IntegerValue{kind, 42}).value,
      RealValue::FromInteger(kind, IntegerValue{kind, 21}).value};
  v.print(os);

  const char *results[]{"(4.2e1_2,2.1e1_2)", "(4.2e1_3,2.1e1_3)",
      "(4.2e1_4,2.1e1_4)", "(4.2e1_8,2.1e1_8)", "(4.2e1_10,2.1e1_10)",
      "(4.2e1_16,2.1e1_16)"};
  EXPECT_EQ(results[pos], os.str());
}

} // namespace
