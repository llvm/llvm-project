//===-- lib/Evaluate/complex-value.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Evaluate/complex-value.h"
#include "flang/Common/idioms.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

namespace Fortran::evaluate::value {

// ============================================================================
// ComplexValue out-of-line definitions.
// ============================================================================

int ComplexValue::kind() const {
  if (IsMonostate()) {
    llvm_unreachable("uninitialized value has not a kind assigned");
  }
  return re_.kind();
}

int ComplexValue::bits() const { return REAL().bits() + AIMAG().bits(); }

ComplexValue ComplexValue::Zero(int kind) {
  RealValue zero{RealValue::Zero(kind)};
  return ComplexValue{zero, zero};
}

ComplexValue ComplexValue::FromRawBytes(const void *raw, std::size_t bytes) {
  CHECK(bytes % 2 == 0);
  std::size_t partBytes{bytes / 2};
  const char *data{static_cast<const char *>(raw)};
  RealValue realPart{RealValue::FromRawBytes(data, partBytes)};
  RealValue imagPart{RealValue::FromRawBytes(data + partBytes, partBytes)};
  return {realPart, imagPart};
}

bool ComplexValue::StoreRawBytes(void *to, std::size_t strideBytes) const {
  CHECK(strideBytes % 2 == 0);
  if (IsMonostate()) {
    llvm_unreachable("uninitialized value");
  }
  auto *bytes{static_cast<char *>(to)};
  std::size_t partBytes{strideBytes / 2};
  bool changed{false};
  changed = re_.StoreRawBytes(bytes, partBytes) || changed;
  changed = im_.StoreRawBytes(bytes + partBytes, partBytes) || changed;
  return changed;
}

bool ComplexValue::IsZero() const { return re_.IsZero() && im_.IsZero(); }

bool ComplexValue::IsInfinite() const {
  return re_.IsInfinite() || im_.IsInfinite();
}

bool ComplexValue::IsNotANumber() const {
  return re_.IsNotANumber() || im_.IsNotANumber();
}

bool ComplexValue::IsSignalingNaN() const {
  return re_.IsSignalingNaN() || im_.IsSignalingNaN();
}

bool ComplexValue::operator==(const ComplexValue &y) const {
  if (IsMonostate() || y.IsMonostate()) {
    return IsMonostate() && y.IsMonostate();
  }
  return re_ == y.re_ && im_ == y.im_;
}

bool ComplexValue::Equals(const ComplexValue &y) const {
  if (IsMonostate()) {
    llvm_unreachable("uncomparable values");
  }
  return re_.Compare(y.re_) == Relation::Equal &&
      im_.Compare(y.im_) == Relation::Equal;
}

RealValue ComplexValue::REAL() const { return re_; }

RealValue ComplexValue::AIMAG() const { return im_; }

ComplexValue ComplexValue::CONJG() const {
  if (IsMonostate()) {
    return ComplexValue{};
  }
  return ComplexValue{re_, im_.Negate()};
}

ComplexValue ComplexValue::Negate() const {
  if (IsMonostate()) {
    return ComplexValue{};
  }
  return ComplexValue{re_.Negate(), im_.Negate()};
}

ComplexValue ComplexValue::FlushSubnormalToZero() const {
  if (IsMonostate()) {
    return ComplexValue{};
  }
  return ComplexValue{re_.FlushSubnormalToZero(), im_.FlushSubnormalToZero()};
}

ValueWithRealFlags<ComplexValue> ComplexValue::Add(
    const ComplexValue &y, Rounding rounding) const {
  if (IsMonostate()) {
    llvm_unreachable("unsupported operation over uninitialized value");
  }
  RealFlags flags;
  RealValue reSum{re_.Add(y.re_, rounding).AccumulateFlags(flags)};
  RealValue imSum{im_.Add(y.im_, rounding).AccumulateFlags(flags)};
  return {ComplexValue{reSum, imSum}, flags};
}

ValueWithRealFlags<ComplexValue> ComplexValue::Subtract(
    const ComplexValue &y, Rounding rounding) const {
  if (IsMonostate()) {
    llvm_unreachable("unsupported operation over uninitialized value");
  }
  RealFlags flags;
  RealValue reDiff{re_.Subtract(y.re_, rounding).AccumulateFlags(flags)};
  RealValue imDiff{im_.Subtract(y.im_, rounding).AccumulateFlags(flags)};
  return {ComplexValue{reDiff, imDiff}, flags};
}

ValueWithRealFlags<ComplexValue> ComplexValue::Multiply(
    const ComplexValue &y, Rounding rounding) const {
  if (IsMonostate()) {
    llvm_unreachable("unsupported operation over uninitialized value");
  }
  // (a + ib)*(c + id) -> ac - bd + i(ad + bc)
  RealFlags flags;
  RealValue ac{re_.Multiply(y.re_, rounding).AccumulateFlags(flags)};
  RealValue bd{im_.Multiply(y.im_, rounding).AccumulateFlags(flags)};
  RealValue ad{re_.Multiply(y.im_, rounding).AccumulateFlags(flags)};
  RealValue bc{im_.Multiply(y.re_, rounding).AccumulateFlags(flags)};
  RealValue acbd{ac.Subtract(bd, rounding).AccumulateFlags(flags)};
  RealValue adbc{ad.Add(bc, rounding).AccumulateFlags(flags)};
  return {ComplexValue{acbd, adbc}, flags};
}

ValueWithRealFlags<ComplexValue> ComplexValue::Divide(
    const ComplexValue &that, Rounding rounding) const {
  if (IsMonostate()) {
    llvm_unreachable("unsupported operation over uninitialized value");
  }
  // (a + ib)/(c + id) -> [(a+ib)*(c-id)] / [(c+id)*(c-id)]
  //   -> [ac+bd+i(bc-ad)] / (cc+dd)  -- note (cc+dd) is real
  //   -> ((ac+bd)/(cc+dd)) + i((bc-ad)/(cc+dd))
  RealFlags flags;
  RealValue cc{that.re_.Multiply(that.re_, rounding).AccumulateFlags(flags)};
  RealValue dd{that.im_.Multiply(that.im_, rounding).AccumulateFlags(flags)};
  RealValue ccPdd{cc.Add(dd, rounding).AccumulateFlags(flags)};
  if (!flags.test(RealFlag::Overflow) && !flags.test(RealFlag::Underflow)) {
    // den = (cc+dd) did not overflow or underflow; try the naive
    // sequence without scaling to avoid extra roundings.
    RealValue ac{re_.Multiply(that.re_, rounding).AccumulateFlags(flags)};
    RealValue ad{re_.Multiply(that.im_, rounding).AccumulateFlags(flags)};
    RealValue bc{im_.Multiply(that.re_, rounding).AccumulateFlags(flags)};
    RealValue bd{im_.Multiply(that.im_, rounding).AccumulateFlags(flags)};
    RealValue acPbd{ac.Add(bd, rounding).AccumulateFlags(flags)};
    RealValue bcSad{bc.Subtract(ad, rounding).AccumulateFlags(flags)};
    RealValue re{acPbd.Divide(ccPdd, rounding).AccumulateFlags(flags)};
    RealValue im{bcSad.Divide(ccPdd, rounding).AccumulateFlags(flags)};
    if (!flags.test(RealFlag::Overflow) && !flags.test(RealFlag::Underflow)) {
      return {ComplexValue{re, im}, flags};
    }
  }
  // Scale numerator and denominator by d/c (if c>=d) or c/d (if c<d)
  flags.clear();
  RealValue scale; // will be <= 1.0 in magnitude
  bool cGEd{that.re_.ABS().Compare(that.im_.ABS()) != Relation::Less};
  if (cGEd) {
    scale = that.im_.Divide(that.re_, rounding).AccumulateFlags(flags);
  } else {
    scale = that.re_.Divide(that.im_, rounding).AccumulateFlags(flags);
  }
  RealValue den;
  if (cGEd) {
    RealValue dS{scale.Multiply(that.im_, rounding).AccumulateFlags(flags)};
    den = dS.Add(that.re_, rounding).AccumulateFlags(flags);
  } else {
    RealValue cS{scale.Multiply(that.re_, rounding).AccumulateFlags(flags)};
    den = cS.Add(that.im_, rounding).AccumulateFlags(flags);
  }
  RealValue aS{scale.Multiply(re_, rounding).AccumulateFlags(flags)};
  RealValue bS{scale.Multiply(im_, rounding).AccumulateFlags(flags)};
  RealValue re1, im1;
  if (cGEd) {
    re1 = re_.Add(bS, rounding).AccumulateFlags(flags);
    im1 = im_.Subtract(aS, rounding).AccumulateFlags(flags);
  } else {
    re1 = aS.Add(im_, rounding).AccumulateFlags(flags);
    im1 = bS.Subtract(re_, rounding).AccumulateFlags(flags);
  }
  RealValue re{re1.Divide(den, rounding).AccumulateFlags(flags)};
  RealValue im{im1.Divide(den, rounding).AccumulateFlags(flags)};
  return {ComplexValue{re, im}, flags};
}

ValueWithRealFlags<RealValue> ComplexValue::ABS(Rounding rounding) const {
  if (IsMonostate()) {
    return ValueWithRealFlags<RealValue>{};
  }
  return re_.HYPOT(im_, rounding);
}

ComplexValue ComplexValue::ConvertToKind(int kind) const {
  if (IsMonostate()) {
    return ComplexValue{};
  }
  return ComplexValue{re_.ConvertToKind(kind), im_.ConvertToKind(kind)};
}

ValueWithRealFlags<ComplexValue> ComplexValue::FromInteger(
    const IntegerValue &n, int kind, bool isUnsigned, Rounding rounding) {
  if (n.IsMonostate()) {
    return ValueWithRealFlags<ComplexValue>{};
  }
  ValueWithRealFlags<ComplexValue> result;
  result.value.re_ = RealValue::FromInteger(n, kind, isUnsigned, rounding)
                         .AccumulateFlags(result.flags);
  result.value.im_ = RealValue::Zero(kind);
  return result;
}

ValueWithRealFlags<ComplexValue> ComplexValue::KahanSummation(
    const ComplexValue &y, ComplexValue &correction, Rounding rounding) const {
  if (IsMonostate()) {
    return ValueWithRealFlags<ComplexValue>{};
  }
  RealFlags flags;
  RealValue reSum{re_.KahanSummation(y.re_, correction.re_, rounding)
          .AccumulateFlags(flags)};
  RealValue imSum{im_.KahanSummation(y.im_, correction.im_, rounding)
          .AccumulateFlags(flags)};
  return {ComplexValue{reSum, imSum}, flags};
}

std::string ComplexValue::DumpHexadecimal() const {
  if (IsMonostate()) {
    llvm_unreachable("unsupported operation over uninitialized value");
  }
  std::string result{'('};
  result += re_.DumpHexadecimal();
  result += ',';
  result += im_.DumpHexadecimal();
  result += ')';
  return result;
}

llvm::raw_ostream &ComplexValue::AsFortran(
    llvm::raw_ostream &o, int kind) const {
  if (IsMonostate()) {
    o << "0";
    return o;
  }
  re_.AsFortran(o << '(', kind);
  im_.AsFortran(o << ',', kind);
  return o << ')';
}

} // namespace Fortran::evaluate::value
