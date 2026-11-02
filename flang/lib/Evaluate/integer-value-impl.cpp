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
  return WithWordProto(kind, [](auto proto) {
    using T = decltype(proto);
    return FromWord(T{});
  });
}

IntegerValueImpl IntegerValueImpl::FromRawBytes(
    int kind, const void *raw, std::size_t expectedSize) {
  CHECK(expectedSize == IntegerValue::bytesStored(kind));

  return WithWordProto(kind, [&](auto proto) {
    assert(IntegerValue::bytesStored(kind) == sizeof(proto));
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
  if (IsNull()) {
    DIE("default-initialized value representing 0 with unknown "
        "width does not know its kind");
    return 0;
  }
  return WithWord(
      [](const auto &x) -> int { return std::decay_t<decltype(x)>::bits / 8; });
}

int IntegerValueImpl::bits() const {
  if (IsNull()) {
    return 0;
  }
  return WithWord(
      [](const auto &x) -> int { return std::decay_t<decltype(x)>::bits; });
}

bool IntegerValueImpl::IsZero() const {
  if (IsNull()) {
    return true; // uninitialized int representing 0 is zero
  }
  return WithWord([](const auto &x) { return x.IsZero(); });
}

IntegerValueImpl IntegerValueImpl::MASKL(int kind, int places) {
  return WithWordProto(kind, [&](auto proto) {
    using T = decltype(proto);
    return FromWord(T::MASKL(places));
  });
}

IntegerValueImpl IntegerValueImpl::MASKR(int kind, int places) {
  return WithWordProto(kind, [&](auto proto) {
    using T = decltype(proto);
    return FromWord(T::MASKR(places));
  });
}

IntegerValueImpl IntegerValueImpl::HUGE(int kind) {
  return WithWordProto(kind, [&](auto proto) {
    using T = decltype(proto);
    return FromWord(T::HUGE());
  });
}

IntegerValueImpl IntegerValueImpl::Least(int kind) {
  return WithWordProto(kind, [&](auto proto) {
    using T = decltype(proto);
    return FromWord(T::Least());
  });
}

bool IntegerValueImpl::IsNegative() const {
  if (IsNull()) {
    return false; // uninitialized int representing 0 is not negative
  }
  return WithWord([](const auto &x) { return x.IsNegative(); });
}

std::uint64_t IntegerValueImpl::ToUInt64() const {
  if (IsNull()) {
    return 0;
  }
  return WithWord([](const auto &x) { return x.ToUInt64(); });
}

std::int64_t IntegerValueImpl::ToInt64() const {
  if (IsNull()) {
    return 0;
  }
  return WithWord([](const auto &x) { return x.ToInt64(); });
}

Fortran::common::uint128_t IntegerValueImpl::ToUInt128() const {
  if (IsNull()) {
    return 0;
  }
  return WithWord([](const auto &x) {
    return x.template ToUInt<Fortran::common::uint128_t>();
  });
}

Fortran::common::int128_t IntegerValueImpl::ToInt128() const {
  if (IsNull()) {
    return 0;
  }
  return WithWord([](const auto &x) {
    return x.template ToSInt<Fortran::common::int128_t,
        Fortran::common::uint128_t>();
  });
}

Ordering IntegerValueImpl::CompareSigned(const IntegerValueImpl &y) const {
  if (IsNull() && y.IsNull()) {
    // Both are considered to be zero
    return Ordering::Equal;
  } else if (IsNull()) {
    switch (y.CompareToZeroSigned()) {
    case Ordering::Less:
      return Ordering::Greater;
    case Ordering::Greater:
      return Ordering::Less;
    case Ordering::Equal:
      return Ordering::Equal;
    }
  } else if (y.IsNull()) {
    return CompareToZeroSigned();
  }

  return WithWord([&](const auto &x) -> Ordering {
    using T = std::decay_t<decltype(x)>;
    return x.CompareSigned(y.GetWord<T>());
  });
}

Ordering IntegerValueImpl::CompareUnsigned(const IntegerValueImpl &y) const {
  if (IsNull() && y.IsNull()) {
    // Both are considered to be zero
    return Ordering::Equal;
  } else if (IsNull()) {
    return y.IsZero() ? Ordering::Equal : Ordering::Less;
  } else if (y.IsNull()) {
    return IsZero() ? Ordering::Equal : Ordering::Greater;
  }

  return WithWord([&](const auto &x) -> Ordering {
    using T = std::decay_t<decltype(x)>;
    return x.CompareUnsigned(y.GetWord<T>());
  });
}

Ordering IntegerValueImpl::CompareToZeroSigned() const {
  if (IsNull()) {
    DIE("uncomparable ints");
    return Ordering::Equal;
  }
  return WithWord([](const auto &x) { return x.CompareToZeroSigned(); });
}

typename IntegerValueImpl::ValueWithOverflow IntegerValueImpl::Negate() const {
  if (IsNull()) {
    return ValueWithOverflow{}; // negation of uninitialized int 0 is zero
  }
  return WithWord([](const auto &x) -> ValueWithOverflow {
    auto r{x.Negate()};
    return {FromWord(r.value), r.overflow};
  });
}

typename IntegerValueImpl::ValueWithOverflow IntegerValueImpl::ABS() const {
  if (IsNull()) {
    return ValueWithOverflow{}; // absolute of uninitialized int 0 is zero
  }
  return WithWord([](const auto &x) -> ValueWithOverflow {
    auto r{x.ABS()};
    return {FromWord(r.value), r.overflow};
  });
}

typename IntegerValueImpl::ValueWithCarry IntegerValueImpl::AddUnsigned(
    const IntegerValueImpl &y, bool carryIn) const {
  if (IsNull()) {
    DIE("incompatible ints");
    return ValueWithCarry{};
  }
  return WithWord([&](const auto &x) -> ValueWithCarry {
    using T = std::decay_t<decltype(x)>;
    auto r{x.AddUnsigned(y.GetWord<T>(), carryIn)};
    return {FromWord(r.value), r.carry};
  });
}

typename IntegerValueImpl::ValueWithOverflow IntegerValueImpl::AddSigned(
    const IntegerValueImpl &y) const {
  if (IsNull()) {
    DIE("incompatible ints");
    return ValueWithOverflow{};
  }
  return WithWord([&](const auto &x) -> ValueWithOverflow {
    using T = std::decay_t<decltype(x)>;
    auto r{x.AddSigned(y.GetWord<T>())};
    return {FromWord(r.value), r.overflow};
  });
}

typename IntegerValueImpl::ValueWithOverflow IntegerValueImpl::SubtractSigned(
    const IntegerValueImpl &y) const {
  if (IsNull()) {
    DIE("incompatible ints");
    return ValueWithOverflow{};
  }
  return WithWord([&](const auto &x) -> ValueWithOverflow {
    using T = std::decay_t<decltype(x)>;
    auto r{x.SubtractSigned(y.GetWord<T>())};
    return {FromWord(r.value), r.overflow};
  });
}

typename IntegerValueImpl::ValueWithOverflow IntegerValueImpl::DIM(
    const IntegerValueImpl &y) const {
  if (IsNull()) {
    DIE("incompatible ints");
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
  if (IsNull()) {
    return ValueWithOverflow{IntegerValueImpl{}, false};
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
  if (IsNull()) {
    DIE("incompatible ints");
    return Product{};
  }
  return WithWord([&](const auto &x) -> Product {
    using T = std::decay_t<decltype(x)>;
    auto r{x.MultiplySigned(y.GetWord<T>())};
    return {FromWord(r.upper), FromWord(r.lower),
        r.SignedMultiplicationOverflowed()};
  });
}

typename IntegerValueImpl::Product IntegerValueImpl::MultiplyUnsigned(
    const IntegerValueImpl &y) const {
  if (IsNull()) {
    DIE("incompatible ints");
    return Product{};
  }
  return WithWord([&](const auto &x) -> Product {
    using T = std::decay_t<decltype(x)>;
    auto r{x.MultiplyUnsigned(y.GetWord<T>())};
    return {FromWord(r.upper), FromWord(r.lower), false};
  });
}

typename IntegerValueImpl::QuotientWithRemainder IntegerValueImpl::DivideSigned(
    const IntegerValueImpl &y) const {
  if (IsNull()) {
    DIE("incompatible ints");
    return QuotientWithRemainder{};
  }
  return WithWord([&](const auto &x) -> QuotientWithRemainder {
    using T = std::decay_t<decltype(x)>;
    auto r{x.DivideSigned(y.GetWord<T>())};
    return {FromWord(r.quotient), FromWord(r.remainder), r.divisionByZero,
        r.overflow};
  });
}

typename IntegerValueImpl::QuotientWithRemainder
IntegerValueImpl::DivideUnsigned(const IntegerValueImpl &y) const {
  if (IsNull()) {
    DIE("incompatible ints");
    return QuotientWithRemainder{};
  }
  return WithWord([&](const auto &x) -> QuotientWithRemainder {
    using T = std::decay_t<decltype(x)>;
    auto r{x.DivideUnsigned(y.GetWord<T>())};
    return {FromWord(r.quotient), FromWord(r.remainder), r.divisionByZero,
        r.overflow};
  });
}

typename IntegerValueImpl::ValueWithOverflow IntegerValueImpl::MODULO(
    const IntegerValueImpl &y) const {
  if (IsNull()) {
    DIE("incompatible ints");
    return ValueWithOverflow{};
  }
  return WithWord([&](const auto &x) -> ValueWithOverflow {
    using T = std::decay_t<decltype(x)>;
    auto r{x.MODULO(y.GetWord<T>())};
    return {FromWord(r.value), r.overflow};
  });
}

typename IntegerValueImpl::PowerWithErrors IntegerValueImpl::Power(
    const IntegerValueImpl &e) const {
  if (IsNull()) {
    DIE("incompatible ints");
    return PowerWithErrors{};
  }
  return WithWord([&](const auto &x) -> PowerWithErrors {
    using T = std::decay_t<decltype(x)>;
    auto r{x.Power(e.GetWord<T>())};
    return {FromWord(r.power), r.divisionByZero, r.overflow, r.zeroToZero};
  });
}

IntegerValueImpl IntegerValueImpl::NOT() const {
  if (IsNull()) {
    DIE("incompatible int");
    return IntegerValueImpl{};
  }
  return WithWord([](const auto &x) { return FromWord(x.NOT()); });
}

IntegerValueImpl IntegerValueImpl::IAND(const IntegerValueImpl &y) const {
  if (IsNull()) {
    DIE("incompatible ints");
    return IntegerValueImpl{};
  }
  return WithWord([&](const auto &x) {
    using T = std::decay_t<decltype(x)>;
    return FromWord(x.IAND(y.GetWord<T>()));
  });
}

IntegerValueImpl IntegerValueImpl::IOR(const IntegerValueImpl &y) const {
  if (IsNull()) {
    DIE("incompatible ints");
    return IntegerValueImpl{};
  }
  return WithWord([&](const auto &x) {
    using T = std::decay_t<decltype(x)>;
    return FromWord(x.IOR(y.GetWord<T>()));
  });
}

IntegerValueImpl IntegerValueImpl::IEOR(const IntegerValueImpl &y) const {
  if (IsNull()) {
    DIE("incompatible ints");
    return IntegerValueImpl{};
  }
  return WithWord([&](const auto &x) {
    using T = std::decay_t<decltype(x)>;
    return FromWord(x.IEOR(y.GetWord<T>()));
  });
}

IntegerValueImpl IntegerValueImpl::MERGE_BITS(
    const IntegerValueImpl &y, const IntegerValueImpl &mask) const {
  if (IsNull()) {
    DIE("incompatible ints");
    return IntegerValueImpl{};
  }
  return WithWord([&](const auto &x) {
    using T = std::decay_t<decltype(x)>;
    return FromWord(x.MERGE_BITS(y.GetWord<T>(), mask.GetWord<T>()));
  });
}

IntegerValueImpl IntegerValueImpl::SHIFTL(int count) const {
  if (IsNull()) {
    DIE("incompatible ints");
    return IntegerValueImpl{};
  }
  return WithWord([&](const auto &x) { return FromWord(x.SHIFTL(count)); });
}

IntegerValueImpl IntegerValueImpl::SHIFTR(int count) const {
  if (IsNull()) {
    DIE("incompatible ints");
    return IntegerValueImpl{};
  }
  return WithWord([&](const auto &x) { return FromWord(x.SHIFTR(count)); });
}

IntegerValueImpl IntegerValueImpl::SHIFTA(int count) const {
  if (IsNull()) {
    DIE("incompatible ints");
    return IntegerValueImpl{};
  }
  return WithWord([&](const auto &x) { return FromWord(x.SHIFTA(count)); });
}

IntegerValueImpl IntegerValueImpl::ISHFTC(int count, int size) const {
  if (IsNull()) {
    DIE("incompatible ints");
    return IntegerValueImpl{};
  }
  return WithWord([&](const auto &x) {
    using T = std::decay_t<decltype(x)>;
    return FromWord(x.ISHFTC(count, size <= 0 ? T::bits : size));
  });
}

IntegerValueImpl IntegerValueImpl::IBITS(int pos, int size) const {
  if (IsNull()) {
    DIE("incompatible ints");
    return IntegerValueImpl{};
  }
  return WithWord([&](const auto &x) { return FromWord(x.IBITS(pos, size)); });
}

IntegerValueImpl IntegerValueImpl::IBSET(int pos) const {
  if (IsNull()) {
    DIE("incompatible ints");
    return IntegerValueImpl{};
  }
  return WithWord([&](const auto &x) { return FromWord(x.IBSET(pos)); });
}

IntegerValueImpl IntegerValueImpl::IBCLR(int pos) const {
  if (IsNull()) {
    DIE("incompatible ints");
    return IntegerValueImpl{};
  }
  return WithWord([&](const auto &x) { return FromWord(x.IBCLR(pos)); });
}

IntegerValueImpl IntegerValueImpl::DSHIFTL(
    const IntegerValueImpl &fill, int count) const {
  if (IsNull()) {
    DIE("incompatible ints");
    return IntegerValueImpl{};
  }
  // DSHIFTL(I,J) shifts I:J left; the second argument is the right fill.
  return WithWord([&](const auto &x) {
    using T = std::decay_t<decltype(x)>;
    return FromWord(x.SHIFTLWithFill(fill.GetWord<T>(), count));
  });
}

IntegerValueImpl IntegerValueImpl::DSHIFTR(
    const IntegerValueImpl &v2, int count) const {
  if (IsNull()) {
    DIE("incompatible ints");
    return IntegerValueImpl{};
  }
  // DSHIFTR(I,J) shifts I:J right; the *first* argument (this) is the left
  // fill, and the receiver of the shift is v2 (mirrors value::Integer's
  // DSHIFTR, whose *this is the shifted operand and whose argument is the
  // fill).
  return v2.WithWord([&](const auto &x2) {
    using T = std::decay_t<decltype(x2)>;
    return FromWord(x2.SHIFTRWithFill(GetWord<T>(), count));
  });
}

bool IntegerValueImpl::BTEST(int pos) const {
  if (IsNull()) {
    return false; // uninitialized int representing 0 has no bits set
  }
  return WithWord([&](const auto &x) { return x.BTEST(pos); });
}

int IntegerValueImpl::LEADZ() const {
  if (IsNull()) {
    DIE("incompatible ints");
    return 0;
  }
  return WithWord([](const auto &x) { return x.LEADZ(); });
}

int IntegerValueImpl::TRAILZ() const {
  if (IsNull()) {
    DIE("incompatible ints");
    return 0;
  }
  return WithWord([](const auto &x) { return x.TRAILZ(); });
}

int IntegerValueImpl::POPCNT() const {
  if (IsNull()) {
    return 0; // uninitialized int representing 0 has no bits set
  }
  return WithWord([](const auto &x) { return x.POPCNT(); });
}

bool IntegerValueImpl::POPPAR() const {
  if (IsNull()) {
    DIE("incompatible ints");
    return false;
  }
  return WithWord([](const auto &x) { return x.POPPAR(); });
}

typename IntegerValueImpl::ValueWithOverflow IntegerValueImpl::ConvertSigned(
    int toKind, const IntegerValueImpl &from) {
  if (from.IsNull()) {
    // Now we know the kind
    return {Zero(toKind), false};
  }
  return from.WithWord([&](const auto &x) -> ValueWithOverflow {
    using S = std::decay_t<decltype(x)>;
    return WithWordProto(toKind, [&](auto proto) -> ValueWithOverflow {
      using T = decltype(proto);
      auto r{T::template ConvertSigned<S>(x)};
      return {FromWord(r.value), r.overflow};
    });
  });
}

typename IntegerValueImpl::ValueWithOverflow IntegerValueImpl::ConvertUnsigned(
    int toKind, const IntegerValueImpl &from) {
  if (from.IsNull()) {
    // Now we know the kind
    return {Zero(toKind), false};
  }
  return from.WithWord([&](const auto &x) -> ValueWithOverflow {
    using S = std::decay_t<decltype(x)>;
    return WithWordProto(toKind, [&](auto proto) -> ValueWithOverflow {
      using T = decltype(proto);
      auto r{T::template ConvertUnsigned<S>(x)};
      return {FromWord(r.value), r.overflow};
    });
  });
}

typename IntegerValueImpl::ValueWithOverflow IntegerValueImpl::Read(
    int kind, const char *&pp, int base, bool isSigned) {
  return WithWordProto(kind, [&](auto proto) -> ValueWithOverflow {
    using T = decltype(proto);
    auto r{T::Read(pp, base, isSigned)};
    return {FromWord(r.value), r.overflow};
  });
}

std::string IntegerValueImpl::SignedDecimal() const {
  if (IsNull()) {
    return "0";
  }
  return WithWord([](const auto &x) { return x.SignedDecimal(); });
}

std::string IntegerValueImpl::UnsignedDecimal() const {
  if (IsNull()) {
    return "0";
  }
  return WithWord([](const auto &x) { return x.UnsignedDecimal(); });
}

std::string IntegerValueImpl::Hexadecimal() const {
  if (IsNull()) {
    return "0";
  }
  return WithWord([](const auto &x) { return x.Hexadecimal(); });
}

void IntegerValueImpl::StoreRawBytes(
    void *dst, size_t expectedSize, bool *changed) const {
  CHECK(expectedSize == bytesStored());

  WithWord([dst, changed, bytesStored = bytesStored()](auto w) {
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
