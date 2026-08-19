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

void ComplexValue::print(llvm::raw_ostream &os) const {
  AsFortran(os, static_cast<int>(kind()));
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void ComplexValue::dump() const {
  print(llvm::errs());
  llvm::errs() << '\n';
}
#endif

ValueWithRealFlags<ComplexValue> ComplexValue::FromInteger(
    KindsEnum kind, const IntegerValue &n, bool isUnsigned, Rounding rounding) {
  CHECK(!n.IsMonostate());

  ValueWithRealFlags<ComplexValue> result;
  result.value.re_ = RealValue::FromInteger(kind, n, isUnsigned, rounding)
                         .AccumulateFlags(result.flags);
  result.value.im_ = RealValue::Zero(kind);
  return result;
}

ValueWithRealFlags<ComplexValue> ComplexValue::Add(
    const ComplexValue &y, Rounding rounding) const {
  CHECK(!IsMonostate());

  RealFlags flags;
  RealValue reSum{re_.Add(y.re_, rounding).AccumulateFlags(flags)};
  RealValue imSum{im_.Add(y.im_, rounding).AccumulateFlags(flags)};
  return {ComplexValue{reSum, imSum}, flags};
}

ValueWithRealFlags<ComplexValue> ComplexValue::Subtract(
    const ComplexValue &y, Rounding rounding) const {
  CHECK(!IsMonostate());

  RealFlags flags;
  RealValue reDiff{re_.Subtract(y.re_, rounding).AccumulateFlags(flags)};
  RealValue imDiff{im_.Subtract(y.im_, rounding).AccumulateFlags(flags)};
  return {ComplexValue{reDiff, imDiff}, flags};
}

ValueWithRealFlags<ComplexValue> ComplexValue::Multiply(
    const ComplexValue &y, Rounding rounding) const {
  CHECK(!IsMonostate());

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
  CHECK(!IsMonostate());

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

ValueWithRealFlags<ComplexValue> ComplexValue::KahanSummation(
    const ComplexValue &y, ComplexValue &correction, Rounding rounding) const {
  CHECK(!y.IsMonostate());
  CHECK(!correction.IsMonostate());

  RealFlags flags;
  RealValue reSum{re_.KahanSummation(y.re_, correction.re_, rounding)
          .AccumulateFlags(flags)};
  RealValue imSum{im_.KahanSummation(y.im_, correction.im_, rounding)
          .AccumulateFlags(flags)};
  return {ComplexValue{reSum, imSum}, flags};
}

std::string ComplexValue::DumpHexadecimal() const {
  CHECK(!IsMonostate());

  std::string result{'('};
  result += re_.DumpHexadecimal();
  result += ',';
  result += im_.DumpHexadecimal();
  result += ')';
  return result;
}

llvm::raw_ostream &ComplexValue::AsFortran(
    llvm::raw_ostream &o, int kind) const {
  CHECK(!IsMonostate());

  re_.AsFortran(o << '(', kind);
  im_.AsFortran(o << ',', kind);
  return o << ')';
}

void ComplexValue::StoreRawBytes(
    void *dst, [[maybe_unused]] size_t expectedSize, bool *changed) const {
  CHECK(!IsMonostate());
  CHECK(re_.bits() == im_.bits());
  CHECK(expectedSize == re_.bytesStored() + im_.bytesStored());

  re_.StoreRawBytes(dst, re_.bytesStored(), changed);
  im_.StoreRawBytes(
      static_cast<char *>(dst) + re_.bytesStored(), im_.bytesStored(), changed);
}

ComplexValue ComplexValue::FromRawBytes(
    KindsEnum kind, const void *raw, std::size_t expectedSize) {
  CHECK(expectedSize == static_cast<size_t>(-1) ||
      expectedSize == bytesStored(kind));
  std::size_t partBytes{RealValue::bytesStored(kind)};
  const char *data{static_cast<const char *>(raw)};
  RealValue realPart{RealValue::FromRawBytes(kind, data, partBytes)};
  RealValue imagPart{
      RealValue::FromRawBytes(kind, data + partBytes, partBytes)};
  return {realPart, imagPart};
}

} // namespace Fortran::evaluate::value
