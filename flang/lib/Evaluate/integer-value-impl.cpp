//===-- lib/Evaluate/integer-value.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "integer-value-impl.h"
#include "flang/Evaluate/integer-value.h"
#include <new>

namespace Fortran::evaluate::value {

IntegerValueImpl IntegerValueImpl::Zero(int kind) {
  return withWordProto(kind, [](auto proto) {
    using T = decltype(proto);
    return FromWord(T{});
  });
}

IntegerValueImpl IntegerValueImpl::FromRawBytes(
    int kind, const void *raw, std::size_t expectedSize) {
  CHECK(
      expectedSize == IntegerValue::bytesStored(static_cast<KindsEnum>(kind)));

  return withWordProto(kind, [&](auto proto) {
    assert(IntegerValue::bytesStored(static_cast<KindsEnum>(kind)) ==
        sizeof(proto));
    std::decay_t<decltype(proto)> t{};
    memcpy(&t, raw, sizeof(proto));
    return FromWord(t);
  });
}

void IntegerValueImpl::print(llvm::raw_ostream &os) const {
  os << SignedDecimal() << '_' << kind();
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void IntegerValueImpl::dump() const {
  print(llvm::errs());
  llvm::errs() << '\n';
}
#endif

int IntegerValueImpl::kind() const {
  if (IsMonostate()) {
    llvm_unreachable("default-initialized value representing 0 with unknown "
                     "width does not know its kind");
    return 0;
  }
  return withWord(
      [](const auto &x) -> int { return std::decay_t<decltype(x)>::bits / 8; });
}

int IntegerValueImpl::bits() const {
  if (IsMonostate()) {
    return 0;
  }
  return withWord(
      [](const auto &x) -> int { return std::decay_t<decltype(x)>::bits; });
}

bool IntegerValueImpl::IsZero() const {
  if (IsMonostate()) {
    return true; // uninitialized int representing 0 is zero
  }
  return withWord([](const auto &x) { return x.IsZero(); });
}

bool IntegerValueImpl::operator==(const IntegerValueImpl &y) const {
  if (IsMonostate() && y.IsMonostate()) {
    return true;
  }
  if (IsMonostate() != y.IsMonostate() || bits() != y.bits()) {
    llvm_unreachable("uncomparable integers");
    return false;
  }
  return withWord([&](const auto &x) -> bool {
    using T = std::decay_t<decltype(x)>;
    return x == std::get<T>(y.storage_);
  });
}

IntegerValueImpl IntegerValueImpl::MASKL(int kind, int places) {
  return withWordProto(kind, [&](auto proto) {
    using T = decltype(proto);
    return FromWord(T::MASKL(places));
  });
}

IntegerValueImpl IntegerValueImpl::MASKR(int kind, int places) {
  return withWordProto(kind, [&](auto proto) {
    using T = decltype(proto);
    return FromWord(T::MASKR(places));
  });
}

IntegerValueImpl IntegerValueImpl::HUGE(int kind) {
  return withWordProto(kind, [&](auto proto) {
    using T = decltype(proto);
    return FromWord(T::HUGE());
  });
}

IntegerValueImpl IntegerValueImpl::Least(int kind) {
  return withWordProto(kind, [&](auto proto) {
    using T = decltype(proto);
    return FromWord(T::Least());
  });
}

bool IntegerValueImpl::IsNegative() const {
  if (IsMonostate()) {
    return false; // uninitialized int representing 0 is not negative
  }
  return withWord([](const auto &x) { return x.IsNegative(); });
}

std::uint64_t IntegerValueImpl::ToUInt64() const {
  if (IsMonostate()) {
    return 0;
  }
  return withWord([](const auto &x) { return x.ToUInt64(); });
}

std::int64_t IntegerValueImpl::ToInt64() const {
  if (IsMonostate()) {
    return 0;
  }
  return withWord([](const auto &x) { return x.ToInt64(); });
}

Fortran::common::uint128_t IntegerValueImpl::ToUInt128() const {
  if (IsMonostate()) {
    return 0;
  }
  return withWord([](const auto &x) {
    return x.template ToUInt<Fortran::common::uint128_t>();
  });
}

Fortran::common::int128_t IntegerValueImpl::ToInt128() const {
  if (IsMonostate()) {
    return 0;
  }
  return withWord([](const auto &x) {
    return x.template ToSInt<Fortran::common::int128_t,
        Fortran::common::uint128_t>();
  });
}

Ordering IntegerValueImpl::CompareSigned(const IntegerValueImpl &y) const {
  if (IsMonostate()) {
    llvm_unreachable("uncomparable ints");
    return Ordering::Equal;
  }
  return withWord([&](const auto &x) -> Ordering {
    using T = std::decay_t<decltype(x)>;
    return x.CompareSigned(Coerce<T>(y));
  });
}

Ordering IntegerValueImpl::CompareUnsigned(const IntegerValueImpl &y) const {
  if (IsMonostate()) {
    llvm_unreachable("uncomparable ints; cast bitwidth first");
    return Ordering::Equal;
  }
  return withWord([&](const auto &x) -> Ordering {
    using T = std::decay_t<decltype(x)>;
    return x.CompareUnsigned(Coerce<T>(y));
  });
}

Ordering IntegerValueImpl::CompareToZeroSigned() const {
  if (IsMonostate()) {
    llvm_unreachable("uncomparable ints");
    return Ordering::Equal;
  }
  return withWord([](const auto &x) { return x.CompareToZeroSigned(); });
}

typename IntegerValueImpl::ValueWithOverflow IntegerValueImpl::Negate() const {
  if (IsMonostate()) {
    return ValueWithOverflow{}; // negation of uninitialized int 0 is zero
  }
  return withWord([](const auto &x) -> ValueWithOverflow {
    auto r{x.Negate()};
    return {FromWord(r.value), r.overflow};
  });
}

typename IntegerValueImpl::ValueWithOverflow IntegerValueImpl::ABS() const {
  if (IsMonostate()) {
    return ValueWithOverflow{}; // absolute of uninitialized int 0 is zero
  }
  return withWord([](const auto &x) -> ValueWithOverflow {
    auto r{x.ABS()};
    return {FromWord(r.value), r.overflow};
  });
}

typename IntegerValueImpl::ValueWithCarry IntegerValueImpl::AddUnsigned(
    const IntegerValueImpl &y, bool carryIn) const {
  if (IsMonostate()) {
    llvm_unreachable("incompatiable ints");
    return ValueWithCarry{};
  }
  return withWord([&](const auto &x) -> ValueWithCarry {
    using T = std::decay_t<decltype(x)>;
    auto r{x.AddUnsigned(Coerce<T>(y), carryIn)};
    return {FromWord(r.value), r.carry};
  });
}

typename IntegerValueImpl::ValueWithOverflow IntegerValueImpl::AddSigned(
    const IntegerValueImpl &y) const {
  if (IsMonostate()) {
    llvm_unreachable("incompatible ints");
    return ValueWithOverflow{};
  }
  return withWord([&](const auto &x) -> ValueWithOverflow {
    using T = std::decay_t<decltype(x)>;
    auto r{x.AddSigned(Coerce<T>(y))};
    return {FromWord(r.value), r.overflow};
  });
}

typename IntegerValueImpl::ValueWithOverflow IntegerValueImpl::SubtractSigned(
    const IntegerValueImpl &y) const {
  if (IsMonostate()) {
    llvm_unreachable("incompatiable ints");
    return ValueWithOverflow{};
  }
  return withWord([&](const auto &x) -> ValueWithOverflow {
    using T = std::decay_t<decltype(x)>;
    auto r{x.SubtractSigned(Coerce<T>(y))};
    return {FromWord(r.value), r.overflow};
  });
}

typename IntegerValueImpl::ValueWithOverflow IntegerValueImpl::DIM(
    const IntegerValueImpl &y) const {
  if (IsMonostate()) {
    llvm_unreachable("incompatiable ints");
    return ValueWithOverflow{};
  }
  // DIM(X,Y) = MAX(X-Y, 0)
  if (CompareSigned(y) != Ordering::Greater) {
    return {Zero(kind()), false};
  }
  return SubtractSigned(y);
}

typename IntegerValueImpl::ValueWithOverflow IntegerValueImpl::SIGN(
    const IntegerValueImpl &sign) const {
  if (IsMonostate()) {
    llvm_unreachable("incompatiable ints");
    return ValueWithOverflow{};
  }
  bool toNegative{sign.IsNegative()};
  if (toNegative == IsNegative()) {
    return {*this, false};
  }
  if (toNegative) {
    return Negate();
  }
  return ABS();
}

typename IntegerValueImpl::Product IntegerValueImpl::MultiplySigned(
    const IntegerValueImpl &y) const {
  if (IsMonostate()) {
    llvm_unreachable("incompatiable ints");
    return Product{};
  }
  return withWord([&](const auto &x) -> Product {
    using T = std::decay_t<decltype(x)>;
    auto r{x.MultiplySigned(Coerce<T>(y))};
    return {FromWord(r.upper), FromWord(r.lower),
        r.SignedMultiplicationOverflowed()};
  });
}

typename IntegerValueImpl::Product IntegerValueImpl::MultiplyUnsigned(
    const IntegerValueImpl &y) const {
  if (IsMonostate()) {
    llvm_unreachable("incompatiable ints");
    return Product{};
  }
  return withWord([&](const auto &x) -> Product {
    using T = std::decay_t<decltype(x)>;
    auto r{x.MultiplyUnsigned(Coerce<T>(y))};
    return {FromWord(r.upper), FromWord(r.lower), false};
  });
}

typename IntegerValueImpl::QuotientWithRemainder IntegerValueImpl::DivideSigned(
    const IntegerValueImpl &y) const {
  if (IsMonostate()) {
    llvm_unreachable("incompatiable ints");
    return QuotientWithRemainder{};
  }
  return withWord([&](const auto &x) -> QuotientWithRemainder {
    using T = std::decay_t<decltype(x)>;
    auto r{x.DivideSigned(Coerce<T>(y))};
    return {FromWord(r.quotient), FromWord(r.remainder), r.divisionByZero,
        r.overflow};
  });
}

typename IntegerValueImpl::QuotientWithRemainder
IntegerValueImpl::DivideUnsigned(const IntegerValueImpl &y) const {
  if (IsMonostate()) {
    llvm_unreachable("incompatiable ints");
    return QuotientWithRemainder{};
  }
  return withWord([&](const auto &x) -> QuotientWithRemainder {
    using T = std::decay_t<decltype(x)>;
    auto r{x.DivideUnsigned(Coerce<T>(y))};
    return {FromWord(r.quotient), FromWord(r.remainder), r.divisionByZero,
        r.overflow};
  });
}

typename IntegerValueImpl::ValueWithOverflow IntegerValueImpl::MODULO(
    const IntegerValueImpl &y) const {
  if (IsMonostate()) {
    llvm_unreachable("incompatiable ints");
    return ValueWithOverflow{};
  }
  return withWord([&](const auto &x) -> ValueWithOverflow {
    using T = std::decay_t<decltype(x)>;
    auto r{x.MODULO(Coerce<T>(y))};
    return {FromWord(r.value), r.overflow};
  });
}

typename IntegerValueImpl::PowerWithErrors IntegerValueImpl::Power(
    const IntegerValueImpl &e) const {
  if (IsMonostate()) {
    llvm_unreachable("incompatiable ints");
    return PowerWithErrors{};
  }
  return withWord([&](const auto &x) -> PowerWithErrors {
    using T = std::decay_t<decltype(x)>;
    auto r{x.Power(Coerce<T>(e))};
    return {FromWord(r.power), r.divisionByZero, r.overflow, r.zeroToZero};
  });
}

IntegerValueImpl IntegerValueImpl::NOT() const {
  if (IsMonostate()) {
    llvm_unreachable("incompatible ints");
    return IntegerValueImpl{};
  }
  return withWord([](const auto &x) { return FromWord(x.NOT()); });
}

IntegerValueImpl IntegerValueImpl::IAND(const IntegerValueImpl &y) const {
  if (IsMonostate()) {
    llvm_unreachable("incompatiable ints");
    return IntegerValueImpl{};
  }
  return withWord([&](const auto &x) {
    using T = std::decay_t<decltype(x)>;
    return FromWord(x.IAND(Coerce<T>(y)));
  });
}

IntegerValueImpl IntegerValueImpl::IOR(const IntegerValueImpl &y) const {
  if (IsMonostate()) {
    llvm_unreachable("incompatible ints");
    return IntegerValueImpl{};
  }
  return withWord([&](const auto &x) {
    using T = std::decay_t<decltype(x)>;
    return FromWord(x.IOR(Coerce<T>(y)));
  });
}

IntegerValueImpl IntegerValueImpl::IEOR(const IntegerValueImpl &y) const {
  if (IsMonostate()) {
    llvm_unreachable("incompatible ints");
    return IntegerValueImpl{};
  }
  return withWord([&](const auto &x) {
    using T = std::decay_t<decltype(x)>;
    return FromWord(x.IEOR(Coerce<T>(y)));
  });
}

IntegerValueImpl IntegerValueImpl::MERGE_BITS(
    const IntegerValueImpl &y, const IntegerValueImpl &mask) const {
  if (IsMonostate()) {
    llvm_unreachable("incompatible ints");
    return IntegerValueImpl{};
  }
  return withWord([&](const auto &x) {
    using T = std::decay_t<decltype(x)>;
    return FromWord(x.MERGE_BITS(Coerce<T>(y), Coerce<T>(mask)));
  });
}

IntegerValueImpl IntegerValueImpl::SHIFTL(int count) const {
  if (IsMonostate()) {
    llvm_unreachable("incompatible ints");
    return IntegerValueImpl{};
  }
  return withWord([&](const auto &x) { return FromWord(x.SHIFTL(count)); });
}

IntegerValueImpl IntegerValueImpl::SHIFTR(int count) const {
  if (IsMonostate()) {
    llvm_unreachable("incompatible ints");
    return IntegerValueImpl{};
  }
  return withWord([&](const auto &x) { return FromWord(x.SHIFTR(count)); });
}

IntegerValueImpl IntegerValueImpl::SHIFTA(int count) const {
  if (IsMonostate()) {
    llvm_unreachable("incompatible ints");
    return IntegerValueImpl{};
  }
  return withWord([&](const auto &x) { return FromWord(x.SHIFTA(count)); });
}

IntegerValueImpl IntegerValueImpl::ISHFTC(int count, int size) const {
  if (IsMonostate()) {
    llvm_unreachable("incompatible ints");
    return IntegerValueImpl{};
  }
  return withWord([&](const auto &x) {
    using T = std::decay_t<decltype(x)>;
    return FromWord(x.ISHFTC(count, size <= 0 ? T::bits : size));
  });
}

IntegerValueImpl IntegerValueImpl::IBITS(int pos, int size) const {
  if (IsMonostate()) {
    llvm_unreachable("incompatible ints");
    return IntegerValueImpl{};
  }
  return withWord([&](const auto &x) { return FromWord(x.IBITS(pos, size)); });
}

IntegerValueImpl IntegerValueImpl::IBSET(int pos) const {
  if (IsMonostate()) {
    llvm_unreachable("incompatible ints");
    return IntegerValueImpl{};
  }
  return withWord([&](const auto &x) { return FromWord(x.IBSET(pos)); });
}

IntegerValueImpl IntegerValueImpl::IBCLR(int pos) const {
  if (IsMonostate()) {
    llvm_unreachable("incompatible ints");
    return IntegerValueImpl{};
  }
  return withWord([&](const auto &x) { return FromWord(x.IBCLR(pos)); });
}

IntegerValueImpl IntegerValueImpl::DSHIFTL(
    const IntegerValueImpl &fill, int count) const {
  if (IsMonostate()) {
    llvm_unreachable("incompatible ints");
    return IntegerValueImpl{};
  }
  // DSHIFTL(I,J) shifts I:J left; the second argument is the right fill.
  return withWord([&](const auto &x) {
    using T = std::decay_t<decltype(x)>;
    return FromWord(x.SHIFTLWithFill(Coerce<T>(fill), count));
  });
}

IntegerValueImpl IntegerValueImpl::DSHIFTR(
    const IntegerValueImpl &v2, int count) const {
  if (IsMonostate()) {
    llvm_unreachable("incompatible ints");
    return IntegerValueImpl{};
  }
  // DSHIFTR(I,J) shifts I:J right; the *first* argument (this) is the left
  // fill, and the receiver of the shift is v2 (mirrors value::Integer's
  // DSHIFTR, whose *this is the shifted operand and whose argument is the
  // fill).
  return v2.withWord([&](const auto &x2) {
    using T = std::decay_t<decltype(x2)>;
    return FromWord(x2.SHIFTRWithFill(Coerce<T>(*this), count));
  });
}

bool IntegerValueImpl::BTEST(int pos) const {
  if (IsMonostate()) {
    return false; // uninitialized int representing 0 has no bits set
  }
  return withWord([&](const auto &x) { return x.BTEST(pos); });
}

int IntegerValueImpl::LEADZ() const {
  if (IsMonostate()) {
    llvm_unreachable("incompatible ints");
    return 0;
  }
  return withWord([](const auto &x) { return x.LEADZ(); });
}

int IntegerValueImpl::TRAILZ() const {
  if (IsMonostate()) {
    llvm_unreachable("incompatible ints");
    return 0;
  }
  return withWord([](const auto &x) { return x.TRAILZ(); });
}

int IntegerValueImpl::POPCNT() const {
  if (IsMonostate()) {
    return 0; // uninitialized int representing 0 has no bits set
  }
  return withWord([](const auto &x) { return x.POPCNT(); });
}

bool IntegerValueImpl::POPPAR() const {
  if (IsMonostate()) {
    llvm_unreachable("incompatible ints");
    return false;
  }
  return withWord([](const auto &x) { return x.POPPAR(); });
}

typename IntegerValueImpl::ValueWithOverflow IntegerValueImpl::ConvertSigned(
    const IntegerValueImpl &from, int toBits) {
  if (from.IsMonostate()) {
    return {};
  }
  return from.withWord([&](const auto &x) -> ValueWithOverflow {
    using S = std::decay_t<decltype(x)>;
    return withWordProto(toBits / 8, [&](auto proto) -> ValueWithOverflow {
      using T = decltype(proto);
      auto r{T::template ConvertSigned<S>(x)};
      return {FromWord(r.value), r.overflow};
    });
  });
}

typename IntegerValueImpl::ValueWithOverflow IntegerValueImpl::ConvertUnsigned(
    const IntegerValueImpl &from, int toBits) {
  if (from.IsMonostate()) {
    return {};
  }
  return from.withWord([&](const auto &x) -> ValueWithOverflow {
    using S = std::decay_t<decltype(x)>;
    return withWordProto(toBits / 8, [&](auto proto) -> ValueWithOverflow {
      using T = decltype(proto);
      auto r{T::template ConvertUnsigned<S>(x)};
      return {FromWord(r.value), r.overflow};
    });
  });
}

typename IntegerValueImpl::ValueWithOverflow IntegerValueImpl::Read(
    int kind, const char *&pp, int base, bool isSigned) {
  return withWordProto(kind, [&](auto proto) -> ValueWithOverflow {
    using T = decltype(proto);
    auto r{T::Read(pp, base, isSigned)};
    return {FromWord(r.value), r.overflow};
  });
}

std::string IntegerValueImpl::SignedDecimal() const {
  if (IsMonostate()) {
    return "0";
  }
  return withWord([](const auto &x) { return x.SignedDecimal(); });
}

std::string IntegerValueImpl::UnsignedDecimal() const {
  if (IsMonostate()) {
    return "0";
  }
  return withWord([](const auto &x) { return x.UnsignedDecimal(); });
}

std::string IntegerValueImpl::Hexadecimal() const {
  if (IsMonostate()) {
    return "0";
  }
  return withWord([](const auto &x) { return x.Hexadecimal(); });
}

void IntegerValueImpl::StoreRawBytes(
    void *dst, size_t expectedSize, bool *changed) const {
  CHECK(expectedSize == bytesStored());

  withWord([dst, changed, bytesStored = bytesStored()](auto w) {
    assert(bytesStored == sizeof(w));

    if (changed) {
      if (std::memcmp(dst, &w, bytesStored) == 0) {
        return;
      }
      *changed = true;
    }
    std::memcpy(dst, &w, bytesStored);
  });
}

} // namespace Fortran::evaluate::value
