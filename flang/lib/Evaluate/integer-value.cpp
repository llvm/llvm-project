//===-- lib/Evaluate/integer-value.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Evaluate/integer-value.h"
#include "integer-value-impl.h"
#include <new>

namespace Fortran::evaluate::value {
static_assert(sizeof(IntegerValueImpl) == detail::kIntegerObjectSize);
static_assert(alignof(IntegerValueImpl) == detail::kIntegerObjectAlign);
static_assert(sizeof(IntegerValue) == sizeof(IntegerValueImpl));
static_assert(alignof(IntegerValue) == alignof(IntegerValueImpl));

IntegerValue::IntegerValue() { new (this) IntegerValueImpl(); }

IntegerValue::~IntegerValue() { impl().~IntegerValueImpl(); }

IntegerValue::IntegerValue(const IntegerValue &x) {
  new (this) IntegerValueImpl(x.impl());
}

IntegerValue::IntegerValue(IntegerValue &&x) {
  new (this) IntegerValueImpl(std::move(x.impl()));
}

IntegerValue &IntegerValue::operator=(const IntegerValue &x) {
  impl() = x.impl();
  return *this;
}

IntegerValue &IntegerValue::operator=(IntegerValue &&x) {
  impl() = std::move(x.impl());
  return *this;
}

IntegerValue IntegerValue::Zero(KindsEnum kind) {
  return FromImpl(IntegerValueImpl::Zero(static_cast<int>(kind)));
}

bool IntegerValue::IsMonostate() const { return impl().IsMonostate(); }

KindsEnum IntegerValue::kind() const {
  return static_cast<KindsEnum>(impl().kind());
}

void IntegerValue::print(llvm::raw_ostream &os) const { impl().print(os); }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void IntegerValue::dump() const { impl().dump(); }
#endif

bool IntegerValue::operator==(const IntegerValue &y) const {
  return impl() == y.impl();
}

IntegerValue IntegerValue::MASKL(KindsEnum kind, int places) {
  return FromImpl(IntegerValueImpl::MASKL(static_cast<int>(kind), places));
}

IntegerValue IntegerValue::MASKR(KindsEnum kind, int places) {
  return FromImpl(IntegerValueImpl::MASKR(static_cast<int>(kind), places));
}

IntegerValue::ValueWithOverflow IntegerValue::Read(
    KindsEnum kind, const char *&pp, int base, bool isSigned) {
  auto r{IntegerValueImpl::Read(static_cast<int>(kind), pp, base, isSigned)};
  return {FromImpl(std::move(r.value)), r.overflow};
}

IntegerValue::ValueWithOverflow IntegerValue::ConvertUnsigned(
    const IntegerValue &from, int toBits) {
  auto r{IntegerValueImpl::ConvertUnsigned(from.impl(), toBits)};
  return {FromImpl(std::move(r.value)), r.overflow};
}

typename IntegerValue::ValueWithOverflow IntegerValue::ConvertSigned(
    const IntegerValue &from, int toBits) {
  auto r{IntegerValueImpl::ConvertSigned(from.impl(), toBits)};
  return {FromImpl(std::move(r.value)), r.overflow};
}

std::string IntegerValue::UnsignedDecimal() const {
  return impl().UnsignedDecimal();
}

std::string IntegerValue::SignedDecimal() const {
  return impl().SignedDecimal();
}

std::string IntegerValue::Hexadecimal() const { return impl().Hexadecimal(); }

IntegerValue IntegerValue::HUGE(KindsEnum kind) {
  return FromImpl(IntegerValueImpl::HUGE(static_cast<int>(kind)));
}

IntegerValue IntegerValue::Least(KindsEnum kind) {
  return FromImpl(IntegerValueImpl::Least(static_cast<int>(kind)));
}

int IntegerValue::RANGE(KindsEnum kind) {
  return DecimalRange(static_cast<int>(kind) * 8 - 1);
}

int IntegerValue::UnsignedRANGE(KindsEnum kind) {
  return DecimalRange(static_cast<int>(kind) * 8);
}

bool IntegerValue::IsZero() const { return impl().IsZero(); }

bool IntegerValue::IsNegative() const { return impl().IsNegative(); }

int IntegerValue::LEADZ() const { return impl().LEADZ(); }

int IntegerValue::POPCNT() const { return impl().POPCNT(); }

bool IntegerValue::POPPAR() const { return impl().POPPAR(); }

int IntegerValue::TRAILZ() const { return impl().TRAILZ(); }

bool IntegerValue::BTEST(int pos) const { return impl().BTEST(pos); }

Ordering IntegerValue::CompareToZeroSigned() const {
  return impl().CompareToZeroSigned();
}

Ordering IntegerValue::CompareUnsigned(const IntegerValue &y) const {
  return impl().CompareUnsigned(y.impl());
}

Ordering IntegerValue::CompareSigned(const IntegerValue &y) const {
  return impl().CompareSigned(y.impl());
}

std::uint64_t IntegerValue::ToUInt64() const { return impl().ToUInt64(); }

std::int64_t IntegerValue::ToInt64() const { return impl().ToInt64(); }

Fortran::common::uint128_t IntegerValue::ToUInt128() const {
  return impl().ToUInt128();
}

Fortran::common::int128_t IntegerValue::ToInt128() const {
  return impl().ToInt128();
}

IntegerValue IntegerValue::NOT() const { return FromImpl(impl().NOT()); }

typename IntegerValue::ValueWithOverflow IntegerValue::Negate() const {
  auto r{impl().Negate()};
  return {FromImpl(std::move(r.value)), r.overflow};
}
typename IntegerValue::ValueWithOverflow IntegerValue::ABS() const {
  auto r{impl().ABS()};
  return {FromImpl(std::move(r.value)), r.overflow};
}

IntegerValue IntegerValue::SHIFTL(int count) const {
  return FromImpl(impl().SHIFTL(count));
}

IntegerValue IntegerValue::ISHFTC(int count, int size) const {
  return FromImpl(impl().ISHFTC(count, size));
}

IntegerValue IntegerValue::ISHFTC(int count) const {
  return FromImpl(impl().ISHFTC(count));
}

IntegerValue IntegerValue::DSHIFTL(const IntegerValue &fill, int count) const {
  return FromImpl(impl().DSHIFTL(fill.impl(), count));
}

IntegerValue IntegerValue::DSHIFTR(const IntegerValue &v2, int count) const {
  return FromImpl(impl().DSHIFTR(v2.impl(), count));
}

IntegerValue IntegerValue::SHIFTR(int count) const {
  return FromImpl(impl().SHIFTR(count));
}

IntegerValue IntegerValue::SHIFTA(int count) const {
  return FromImpl(impl().SHIFTA(count));
}

IntegerValue IntegerValue::IBCLR(int pos) const {
  return FromImpl(impl().IBCLR(pos));
}

IntegerValue IntegerValue::IBSET(int pos) const {
  return FromImpl(impl().IBSET(pos));
}

IntegerValue IntegerValue::IBITS(int pos, int size) const {
  return FromImpl(impl().IBITS(pos, size));
}

IntegerValue IntegerValue::IAND(const IntegerValue &y) const {
  return FromImpl(impl().IAND(y.impl()));
}

IntegerValue IntegerValue::IOR(const IntegerValue &y) const {
  return FromImpl(impl().IOR(y.impl()));
}

IntegerValue IntegerValue::IEOR(const IntegerValue &y) const {
  return FromImpl(impl().IEOR(y.impl()));
}

IntegerValue IntegerValue::MERGE_BITS(
    const IntegerValue &y, const IntegerValue &mask) const {
  return FromImpl(impl().MERGE_BITS(y.impl(), mask.impl()));
}

typename IntegerValue::ValueWithCarry IntegerValue::AddUnsigned(
    const IntegerValue &y, bool carryIn) const {
  auto r{impl().AddUnsigned(y.impl(), carryIn)};
  return {FromImpl(std::move(r.value)), r.carry};
}

typename IntegerValue::ValueWithOverflow IntegerValue::AddSigned(
    const IntegerValue &y) const {
  auto r{impl().AddSigned(y.impl())};
  return {FromImpl(std::move(r.value)), r.overflow};
}

typename IntegerValue::ValueWithOverflow IntegerValue::SubtractSigned(
    const IntegerValue &y) const {
  auto r{impl().SubtractSigned(y.impl())};
  return {FromImpl(std::move(r.value)), r.overflow};
}

typename IntegerValue::ValueWithOverflow IntegerValue::DIM(
    const IntegerValue &y) const {
  auto r{impl().DIM(y.impl())};
  return {FromImpl(std::move(r.value)), r.overflow};
}

typename IntegerValue::ValueWithOverflow IntegerValue::SIGN(
    const IntegerValue &sign) const {
  auto r{impl().SIGN(sign.impl())};
  return {FromImpl(std::move(r.value)), r.overflow};
}

typename IntegerValue::Product IntegerValue::MultiplyUnsigned(
    const IntegerValue &y) const {
  auto r{impl().MultiplyUnsigned(y.impl())};
  return {
      FromImpl(std::move(r.upper)), FromImpl(std::move(r.lower)), r.overflow};
}

typename IntegerValue::Product IntegerValue::MultiplySigned(
    const IntegerValue &y) const {
  auto r{impl().MultiplySigned(y.impl())};
  return {
      FromImpl(std::move(r.upper)), FromImpl(std::move(r.lower)), r.overflow};
}

typename IntegerValue::QuotientWithRemainder IntegerValue::DivideUnsigned(
    const IntegerValue &y) const {
  auto r{impl().DivideUnsigned(y.impl())};
  return {FromImpl(std::move(r.quotient)), FromImpl(std::move(r.remainder)),
      r.divisionByZero, r.overflow};
}

typename IntegerValue::QuotientWithRemainder IntegerValue::DivideSigned(
    const IntegerValue &y) const {
  auto r{impl().DivideSigned(y.impl())};
  return {FromImpl(std::move(r.quotient)), FromImpl(std::move(r.remainder)),
      r.divisionByZero, r.overflow};
}

typename IntegerValue::ValueWithOverflow IntegerValue::MODULO(
    const IntegerValue &y) const {
  auto r{impl().MODULO(y.impl())};
  return {FromImpl(std::move(r.value)), r.overflow};
}

typename IntegerValue::PowerWithErrors IntegerValue::Power(
    const IntegerValue &e) const {
  auto r{impl().Power(e.impl())};
  return {
      FromImpl(std::move(r.power)), r.divisionByZero, r.overflow, r.zeroToZero};
}

IntegerValue IntegerValue::FromRawBytes(
    KindsEnum kind, const void *raw, std::size_t expectedSize) {
  return FromImpl(IntegerValueImpl::FromRawBytes(
      static_cast<int>(kind), raw, expectedSize));
}

void IntegerValue::StoreRawBytes(void *dst, size_t size, bool *changed) const {
  impl().StoreRawBytes(dst, size, changed);
}

void IntegerValue::ConstructFromIntegral(
    KindsEnum kind, std::uint64_t v, bool isSigned) {
  new (this) IntegerValueImpl(static_cast<int>(kind), v, isSigned);
}

void IntegerValue::ConstructFromIntegral(
    KindsEnum kind, Fortran::common::uint128_t v) {
  new (this) IntegerValueImpl(static_cast<int>(kind), v);
}

IntegerValue IntegerValue::FromImpl(const IntegerValueImpl &x) {
  IntegerValue r;
  r.impl() = x;
  return r;
}

IntegerValue IntegerValue::FromImpl(IntegerValueImpl &&x) {
  IntegerValue r;
  r.impl() = std::move(x);
  return r;
}

} // namespace Fortran::evaluate::value
