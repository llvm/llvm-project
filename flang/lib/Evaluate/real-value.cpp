//===-- lib/Evaluate/real-value.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Evaluate/real-value.h"
#include "real-value-impl.h"
#include "llvm/Support/raw_ostream.h"
#include <new>
#include <string>

namespace Fortran::evaluate::value {
static_assert(sizeof(RealValueImpl) == detail::kRealObjectSize);
static_assert(alignof(RealValueImpl) == detail::kRealObjectAlign);
static_assert(sizeof(RealValue) == sizeof(RealValueImpl));
static_assert(alignof(RealValue) == alignof(RealValueImpl));

RealValue::RealValue() { new (this) RealValueImpl(); }

RealValue::~RealValue() { impl().~RealValueImpl(); }

RealValue::RealValue(const RealValue &x) { new (this) RealValueImpl(x.impl()); }

RealValue::RealValue(RealValue &&x) {
  new (this) RealValueImpl(std::move(x.impl()));
}

RealValue &RealValue::operator=(const RealValue &x) {
  impl() = x.impl();
  return *this;
}

RealValue &RealValue::operator=(RealValue &&x) {
  impl() = std::move(x.impl());
  return *this;
}

RealValue::RealValue(int kind, const Word &w) {
  new (this) RealValueImpl(kind, w);
}

RealValue::RealValue(int kind, double x) { new (this) RealValueImpl(kind, x); }

RealValue RealValue::Zero(int kind) {
  return FromImpl(RealValueImpl::Zero(kind));
}

RealValue RealValue::NegativeZero(int kind) {
  return FromImpl(RealValueImpl::NegativeZero(kind));
}

RealValue RealValue::Infinity(int kind, bool negative) {
  return FromImpl(RealValueImpl::Infinity(kind, negative));
}

bool RealValue::IsMonostate() const { return impl().IsMonostate(); }

int RealValue::kind() const { return impl().kind(); }

void RealValue::print(llvm::raw_ostream &os) const { impl().print(os); }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void RealValue::dump() const { impl().dump(); }
#endif

bool RealValue::operator==(const RealValue &y) const {
  return impl() == y.impl();
}

bool RealValue::IsNegative() const { return impl().IsNegative(); }

bool RealValue::IsNotANumber() const { return impl().IsNotANumber(); }

bool RealValue::IsSignalingNaN() const { return impl().IsSignalingNaN(); }

bool RealValue::IsInfinite() const { return impl().IsInfinite(); }

bool RealValue::IsFinite() const { return impl().IsFinite(); }

bool RealValue::IsZero() const { return impl().IsZero(); }

bool RealValue::IsNormal() const { return impl().IsNormal(); }

RealValue RealValue::ABS() const { return FromImpl(impl().ABS()); }

RealValue RealValue::SetSign(bool toNegative) const {
  return FromImpl(impl().SetSign(toNegative));
}

RealValue RealValue::SIGN(const RealValue &x) const {
  return FromImpl(impl().SIGN(x.impl()));
}

RealValue RealValue::Negate() const { return FromImpl(impl().Negate()); }

Relation RealValue::Compare(const RealValue &y) const {
  return impl().Compare(y.impl());
}

ValueWithRealFlags<RealValue> RealValue::Add(
    const RealValue &y, Rounding rounding) const {
  return FromImpl(impl().Add(y.impl(), rounding));
}
ValueWithRealFlags<RealValue> RealValue::Subtract(
    const RealValue &y, Rounding rounding) const {
  return FromImpl(impl().Subtract(y.impl(), rounding));
}
ValueWithRealFlags<RealValue> RealValue::Multiply(
    const RealValue &y, Rounding rounding) const {
  return FromImpl(impl().Multiply(y.impl(), rounding));
}
ValueWithRealFlags<RealValue> RealValue::Divide(
    const RealValue &y, Rounding rounding) const {
  return FromImpl(impl().Divide(y.impl(), rounding));
}
ValueWithRealFlags<RealValue> RealValue::SQRT(Rounding rounding) const {
  return FromImpl(impl().SQRT(rounding));
}
ValueWithRealFlags<RealValue> RealValue::NEAREST(bool upward) const {
  return FromImpl(impl().NEAREST(upward));
}
ValueWithRealFlags<RealValue> RealValue::HYPOT(
    const RealValue &y, Rounding rounding) const {
  return FromImpl(impl().HYPOT(y.impl(), rounding));
}
ValueWithRealFlags<RealValue> RealValue::DIM(
    const RealValue &y, Rounding rounding) const {
  return FromImpl(impl().DIM(y.impl(), rounding));
}
ValueWithRealFlags<RealValue> RealValue::MOD(
    const RealValue &y, Rounding rounding) const {
  return FromImpl(impl().MOD(y.impl(), rounding));
}
ValueWithRealFlags<RealValue> RealValue::MODULO(
    const RealValue &y, Rounding rounding) const {
  return FromImpl(impl().MODULO(y.impl(), rounding));
}
ValueWithRealFlags<RealValue> RealValue::KahanSummation(
    const RealValue &y, RealValue &correction, Rounding rounding) const {
  return FromImpl(impl().KahanSummation(y.impl(), correction.impl(), rounding));
}

IntegerValue RealValue::EXPONENT() const { return impl().EXPONENT(); }

RealValue RealValue::EPSILON(int kind) {
  return FromImpl(RealValueImpl::EPSILON(kind));
}

RealValue RealValue::HUGE(int kind) {
  return FromImpl(RealValueImpl::HUGE(kind));
}

RealValue RealValue::TINY(int kind) {
  return FromImpl(RealValueImpl::TINY(kind));
}

int RealValue::DIGITS(int kind) { return RealValueImpl::DIGITS(kind); }

int RealValue::PRECISION(int kind) { return RealValueImpl::PRECISION(kind); }

int RealValue::RANGE(int kind) { return RealValueImpl::RANGE(kind); }

int RealValue::MAXEXPONENT(int kind) {
  return RealValueImpl::MAXEXPONENT(kind);
}

int RealValue::MINEXPONENT(int kind) {
  return RealValueImpl::MINEXPONENT(kind);
}

RealValue RealValue::RRSPACING() const { return FromImpl(impl().RRSPACING()); }

RealValue RealValue::SPACING() const { return FromImpl(impl().SPACING()); }

RealValue RealValue::SET_EXPONENT(std::int64_t e) const {
  return FromImpl(impl().SET_EXPONENT(e));
}

RealValue RealValue::FRACTION() const { return FromImpl(impl().FRACTION()); }

ValueWithRealFlags<RealValue> RealValue::SCALE(
    const IntegerValue &by, Rounding rounding) const {
  return FromImpl(impl().SCALE(by, rounding));
}

RealValue RealValue::FlushSubnormalToZero() const {
  return FromImpl(impl().FlushSubnormalToZero());
}

RealValue RealValue::NotANumber(int kind) {
  return FromImpl(RealValueImpl::NotANumber(kind));
}

ValueWithRealFlags<RealValue> RealValue::FromInteger(
    int kind, const IntegerValue &n, bool isUnsigned, Rounding rounding) {
  return FromImpl(RealValueImpl::FromInteger(kind, n, isUnsigned, rounding));
}

ValueWithRealFlags<RealValue> RealValue::ToWholeNumber(
    common::RoundingMode mode) const {
  return FromImpl(impl().ToWholeNumber(mode));
}
ValueWithRealFlags<IntegerValue> RealValue::ToInteger(
    common::RoundingMode mode, int toBits) const {
  return impl().ToInteger(mode, toBits);
}

ValueWithRealFlags<RealValue> RealValue::Convert(
    int kind, const RealValue &from, Rounding rounding) {
  return FromImpl(RealValueImpl::Convert(kind, from.impl(), rounding));
}

IntegerValue RealValue::RawBits() const { return impl().RawBits(); }

int RealValue::Exponent() const { return impl().Exponent(); }

ValueWithRealFlags<RealValue> RealValue::Read(
    int kind, const char *&pp, Rounding rounding) {
  return FromImpl(RealValueImpl::Read(kind, pp, rounding));
}

std::string RealValue::DumpHexadecimal() const {
  return impl().DumpHexadecimal();
}

llvm::raw_ostream &RealValue::AsFortran(
    llvm::raw_ostream &o, int kind, bool minimal) const {
  return impl().AsFortran(o, kind, minimal);
}

RealValue RealValue::FromRawBytes(
    int kind, const void *raw, std::size_t expectedSize) {
  return FromImpl(RealValueImpl::FromRawBytes(kind, raw, expectedSize));
}

void RealValue::StoreRawBytes(void *dst, size_t size, bool *changed) const {
  impl().StoreRawBytes(dst, size, changed);
}

RealValue RealValue::FromImpl(const RealValueImpl &x) {
  RealValue r;
  r.impl() = x;
  return r;
}

RealValue RealValue::FromImpl(RealValueImpl &&x) {
  RealValue r;
  r.impl() = std::move(x);
  return r;
}

ValueWithRealFlags<RealValue> RealValue::FromImpl(
    const ValueWithRealFlags<RealValueImpl> &x) {
  ValueWithRealFlags<RealValue> r;
  r.value.impl() = std::move(x.value);
  r.flags = x.flags;
  return r;
}

ValueWithRealFlags<RealValue> RealValue::FromImpl(
    ValueWithRealFlags<RealValueImpl> &&x) {
  ValueWithRealFlags<RealValue> r;
  r.value.impl() = x.value;
  r.flags = x.flags;
  return r;
}

} // namespace Fortran::evaluate::value
