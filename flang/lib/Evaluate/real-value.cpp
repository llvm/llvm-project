//===-- lib/Evaluate/real-value.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Evaluate/real-value.h"
#include "int-power.h"
#include "real-value-impl.h"
#include "flang/Common/idioms.h"
#include "flang/Decimal/decimal.h"
#include "flang/Evaluate/rounding-bits.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstring>
#include <new>
#include <string>

namespace Fortran::evaluate::value {

RoundingBits::RoundingBits(const IntegerValue &fraction, int rshift) {
  const int fb{fraction.bits()};
  if (rshift > 0 && rshift < fb + 1) {
    guard_ = fraction.BTEST(rshift - 1);
  }
  if (rshift > 1 && rshift < fb + 2) {
    round_ = fraction.BTEST(rshift - 2);
  }
  if (rshift > 2) {
    if (rshift >= fb + 2) {
      sticky_ = !fraction.IsZero();
    } else {
      auto mask{IntegerValue::MASKR(rshift - 2, fraction.kind())};
      sticky_ = !fraction.IAND(mask).IsZero();
    }
  }
}

static_assert(sizeof(RealValueImpl) == detail::kRealObjectSize);
static_assert(alignof(RealValueImpl) <= detail::kRealObjectAlign);
static_assert(sizeof(RealValue) == sizeof(RealValueImpl));

// ============================================================================
// RealValue facade.
// ============================================================================

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
  r.value.impl() = x.value;
  r.flags = x.flags;
  return r;
}

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

RealValue::RealValue(const Word &w, int kind) {
  new (this) RealValueImpl(w, kind);
}

bool RealValue::operator==(const RealValue &y) const {
  return impl() == y.impl();
}

int RealValue::binaryPrecision(int kind) {
  return RealValueImpl::binaryPrecision(kind);
}
int RealValue::isImplicitMSB(int kind) {
  return RealValueImpl::isImplicitMSB(kind);
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
RealValue RealValue::HUGE(int kind) {
  return FromImpl(RealValueImpl::HUGE(kind));
}
RealValue RealValue::EPSILON(int kind) {
  return FromImpl(RealValueImpl::EPSILON(kind));
}
RealValue RealValue::TINY(int kind) {
  return FromImpl(RealValueImpl::TINY(kind));
}
RealValue RealValue::NotANumber(int kind) {
  return FromImpl(RealValueImpl::NotANumber(kind));
}

int RealValue::kind() const { return impl().kind(); }
int RealValue::bits() const { return impl().bits(); }
bool RealValue::IsMonostate() const { return impl().IsMonostate(); }
bool RealValue::IsZero() const { return impl().IsZero(); }
bool RealValue::IsNegative() const { return impl().IsNegative(); }
bool RealValue::IsNotANumber() const { return impl().IsNotANumber(); }
bool RealValue::IsQuietNaN() const { return impl().IsQuietNaN(); }
bool RealValue::IsSignalingNaN() const { return impl().IsSignalingNaN(); }
bool RealValue::IsInfinite() const { return impl().IsInfinite(); }
bool RealValue::IsFinite() const { return impl().IsFinite(); }
bool RealValue::IsNormal() const { return impl().IsNormal(); }
int RealValue::Exponent() const { return impl().Exponent(); }
bool RealValue::StoreRawBytes(void *to, std::size_t bytes) const {
  return impl().StoreRawBytes(to, bytes);
}
RealValue RealValue::FromRawBytes(const void *raw, std::size_t bytes) {
  return FromImpl(RealValueImpl::FromRawBytes(raw, bytes));
}
RealValue RealValue::Zero(int kind) {
  return FromImpl(RealValueImpl::Zero(kind));
}
IntegerValue RealValue::RawBits() const { return impl().RawBits(); }
Relation RealValue::Compare(const RealValue &y) const {
  return impl().Compare(y.impl());
}
RealValue RealValue::ABS() const { return FromImpl(impl().ABS()); }
RealValue RealValue::Negate() const { return FromImpl(impl().Negate()); }
RealValue RealValue::SIGN(const RealValue &x) const {
  return FromImpl(impl().SIGN(x.impl()));
}
RealValue RealValue::SetSign(bool toNegative) const {
  return FromImpl(impl().SetSign(toNegative));
}
RealValue RealValue::FlushSubnormalToZero() const {
  return FromImpl(impl().FlushSubnormalToZero());
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
ValueWithRealFlags<RealValue> RealValue::HYPOT(
    const RealValue &y, Rounding rounding) const {
  return FromImpl(impl().HYPOT(y.impl(), rounding));
}
ValueWithRealFlags<RealValue> RealValue::MOD(
    const RealValue &y, Rounding rounding) const {
  return FromImpl(impl().MOD(y.impl(), rounding));
}
ValueWithRealFlags<RealValue> RealValue::MODULO(
    const RealValue &y, Rounding rounding) const {
  return FromImpl(impl().MODULO(y.impl(), rounding));
}
ValueWithRealFlags<RealValue> RealValue::DIM(
    const RealValue &y, Rounding rounding) const {
  return FromImpl(impl().DIM(y.impl(), rounding));
}
RealValue RealValue::FRACTION() const { return FromImpl(impl().FRACTION()); }
RealValue RealValue::RRSPACING() const { return FromImpl(impl().RRSPACING()); }
RealValue RealValue::SPACING() const { return FromImpl(impl().SPACING()); }
RealValue RealValue::SET_EXPONENT(std::int64_t e) const {
  return FromImpl(impl().SET_EXPONENT(e));
}
ValueWithRealFlags<RealValue> RealValue::NEAREST(bool upward) const {
  return FromImpl(impl().NEAREST(upward));
}
ValueWithRealFlags<RealValue> RealValue::ToWholeNumber(
    common::RoundingMode mode) const {
  return FromImpl(impl().ToWholeNumber(mode));
}
ValueWithRealFlags<IntegerValue> RealValue::ToInteger(
    common::RoundingMode mode, int toBits) const {
  return impl().ToInteger(mode, toBits);
}
ValueWithRealFlags<RealValue> RealValue::SCALE(
    const IntegerValue &by, Rounding rounding) const {
  return FromImpl(impl().SCALE(by, rounding));
}
IntegerValue RealValue::EXPONENT() const { return impl().EXPONENT(); }
ValueWithRealFlags<RealValue> RealValue::KahanSummation(
    const RealValue &y, RealValue &correction, Rounding rounding) const {
  return FromImpl(impl().KahanSummation(y.impl(), correction.impl(), rounding));
}
ValueWithRealFlags<RealValue> RealValue::FromInteger(
    const IntegerValue &n, int kind, bool isUnsigned, Rounding rounding) {
  return FromImpl(RealValueImpl::FromInteger(n, kind, isUnsigned, rounding));
}
ValueWithRealFlags<RealValue> RealValue::Convert(
    const RealValue &from, int kind, Rounding rounding) {
  return FromImpl(RealValueImpl::Convert(from.impl(), kind, rounding));
}
ValueWithRealFlags<RealValue> RealValue::Read(
    const char *&pp, int kind, Rounding rounding) {
  return FromImpl(RealValueImpl::Read(pp, kind, rounding));
}
std::string RealValue::DumpHexadecimal() const {
  return impl().DumpHexadecimal();
}
llvm::raw_ostream &RealValue::AsFortran(
    llvm::raw_ostream &o, int kind, bool minimal) const {
  return impl().AsFortran(o, kind, minimal);
}

// ============================================================================
// RealValueImpl out-of-line definitions.
// ============================================================================

namespace {

template <typename INT> IntegerValue IntegerValueFromFixed(const INT &n) {
  constexpr unsigned bits{INT::bits};
  constexpr unsigned nWords{(bits + 63) / 64};
  std::uint64_t words[nWords]{};
  INT cur{n};
  for (unsigned i = 0; i < nWords; ++i) {
    words[i] = cur.template ToUInt<std::uint64_t>();
    if constexpr (bits > 64) {
      cur = cur.SHIFTR(64);
    }
  }
  return IntegerValue::FromAPInt(
      llvm::APInt(bits, llvm::ArrayRef(words, nWords)));
}

template <typename INT> INT FixedIntegerFromValue(const IntegerValue &v) {
  if (v.IsMonostate()) {
    return {};
  }
  constexpr unsigned bits{INT::bits};
  llvm::APInt z{v.toAPInt().zextOrTrunc(bits)};
  const std::uint64_t *raw{z.getRawData()};
  if constexpr (bits <= 64) {
    return INT{raw[0]};
  } else {
    return INT{raw[0]}.IOR(INT{raw[1]}.SHIFTL(64));
  }
}

} // namespace

template <typename R>
ValueWithRealFlags<R> FromIntegerValue(
    const IntegerValue &n, bool isUnsigned, Rounding rounding) {
  using Fraction = typename R::Fraction;
  bool isNegative{!isUnsigned && n.IsNegative()};
  IntegerValue absN{n};
  if (isNegative) {
    absN = n.Negate().value;
  }
  int leadz{absN.LEADZ()};
  const int absBits{absN.bits()};
  if (leadz >= absBits) {
    return {};
  }
  ValueWithRealFlags<R> result;
  const int exponent{R::exponentBias + absBits - leadz - 1};
  const int bitsNeeded{absBits - (leadz + R::isImplicitMSB)};
  const int bitsLost{bitsNeeded - R::significandBits};
  if (bitsLost <= 0) {
    Fraction fraction{FixedIntegerFromValue<Fraction>(absN)};
    result.flags |= result.value.Normalize(
        isNegative, exponent, fraction.SHIFTL(-bitsLost));
  } else {
    Fraction fraction{FixedIntegerFromValue<Fraction>(absN.SHIFTR(bitsLost))};
    result.flags |= result.value.Normalize(isNegative, exponent, fraction);
    RoundingBits roundingBits{absN, bitsLost};
    result.flags |= result.value.Round(rounding, roundingBits);
  }
  return result;
}

RealValueImpl::RealValueImpl(const Word &w, int kind) {
  realWithKind(kind, [&](auto proto) {
    using R = decltype(proto);
    if (w.IsMonostate()) {
      storage_ = R{};
    } else {
      storage_ = R{FixedIntegerFromValue<typename R::Word>(w)};
    }
  });
}

bool RealValueImpl::operator==(const RealValueImpl &y) const {
  return WithReal([&y](const auto &v1) -> bool {
    return y.WithReal([&v1](const auto &v2) -> bool {
      if constexpr (std::is_same_v<std::decay_t<decltype(v1)>,
                        std::decay_t<decltype(v2)>>) {
        return v1 == v2;
      }
      llvm_unreachable("Uncomparable reals");
    });
  });
}

int RealValueImpl::kind() const {
  if (IsMonostate()) {
    llvm_unreachable("uninitialized value has not a defined kind");
  }
  return WithReal(
      [](const auto &v) { return KindOf<std::decay_t<decltype(v)>>(); });
}

int RealValueImpl::bits() const {
  if (IsMonostate()) {
    return 0;
  }
  return WithReal(
      [](const auto &v) -> int { return std::decay_t<decltype(v)>::bits; });
}

RealValueImpl RealValueImpl::Zero(int kind) {
  RealValueImpl result;
  realWithKind(kind, [&](auto proto) { result.storage_ = decltype(proto){}; });
  return result;
}

// PAPAYA: Remove, not a 1-to-1 relationship
static int RealKindFromByteCount(std::size_t bytes) {
  switch (bytes) {
  case 2:
    return 2;
  case 4:
    return 4;
  case 8:
    return 8;
  case 10:
    return 10;
  case 16:
    return 16;
  default:
    return 8;
  }
}

// PAPAYA: Replace bytes by kind (all types)
RealValueImpl RealValueImpl::FromRawBytes(const void *raw, std::size_t bytes) {
  int kind{RealKindFromByteCount(bytes)};
  return RealValueImpl{IntegerValue::FromRawBytes(raw, kind), kind};
}

IntegerValue RealValueImpl::RawBits() const {
  if (IsMonostate()) {
    return {};
  }
  return WithReal(
      [](const auto &v) { return IntegerValueFromFixed(v.RawBits()); });
}

bool RealValueImpl::StoreRawBytes(void *to, std::size_t bytes) const {
  auto raw{RawBits()};
  std::size_t payloadBytes{
      std::min(bytes, static_cast<std::size_t>((bits() + 7) / 8))};
  bool changed{raw.StoreRawBytes(to, kind())};
  if (payloadBytes < bytes &&
      !std::all_of(
          static_cast<const char *>(to) + payloadBytes,
          static_cast<const char *>(to) + bytes,
          [](char x) { return x == 0; })) {
    std::memset(
        static_cast<char *>(to) + payloadBytes, 0, bytes - payloadBytes);
    changed = true;
  }
  return changed;
}

int RealValueImpl::binaryPrecision(int kind) {
  return realWithKind(
      kind, [](auto p) { return decltype(p)::binaryPrecision; });
}
int RealValueImpl::isImplicitMSB(int kind) {
  return realWithKind(
      kind, [](auto p) -> int { return decltype(p)::isImplicitMSB; });
}
int RealValueImpl::DIGITS(int kind) {
  return realWithKind(kind, [](auto p) { return decltype(p)::DIGITS; });
}
int RealValueImpl::PRECISION(int kind) {
  return realWithKind(kind, [](auto p) { return decltype(p)::PRECISION; });
}
int RealValueImpl::RANGE(int kind) {
  return realWithKind(kind, [](auto p) { return decltype(p)::RANGE; });
}
int RealValueImpl::MAXEXPONENT(int kind) {
  return realWithKind(kind, [](auto p) { return decltype(p)::MAXEXPONENT; });
}
int RealValueImpl::MINEXPONENT(int kind) {
  return realWithKind(kind, [](auto p) { return decltype(p)::MINEXPONENT; });
}
RealValueImpl RealValueImpl::HUGE(int kind) {
  return realWithKind(kind, [](auto p) { return Wrap(decltype(p)::HUGE()); });
}
RealValueImpl RealValueImpl::EPSILON(int kind) {
  return realWithKind(
      kind, [](auto p) { return Wrap(decltype(p)::EPSILON()); });
}
RealValueImpl RealValueImpl::TINY(int kind) {
  return realWithKind(kind, [](auto p) { return Wrap(decltype(p)::TINY()); });
}
RealValueImpl RealValueImpl::NotANumber(int kind) {
  return realWithKind(
      kind, [](auto p) { return Wrap(decltype(p)::NotANumber()); });
}

bool RealValueImpl::IsZero() const {
  if (IsMonostate()) {
    return true;
  }
  return WithReal([](const auto &v) { return v.IsZero(); });
}

bool RealValueImpl::IsNegative() const {
  if (IsMonostate()) {
    return false;
  }
  return WithReal([](const auto &v) { return v.IsNegative(); });
}

bool RealValueImpl::IsNotANumber() const {
  if (IsMonostate()) {
    return false;
  }
  return WithReal([](const auto &v) { return v.IsNotANumber(); });
}

bool RealValueImpl::IsQuietNaN() const {
  if (IsMonostate()) {
    return false;
  }
  return WithReal([](const auto &v) { return v.IsQuietNaN(); });
}

bool RealValueImpl::IsSignalingNaN() const {
  if (IsMonostate()) {
    return false;
  }
  return WithReal([](const auto &v) { return v.IsSignalingNaN(); });
}

bool RealValueImpl::IsInfinite() const {
  if (IsMonostate()) {
    return false;
  }
  return WithReal([](const auto &v) { return v.IsInfinite(); });
}

bool RealValueImpl::IsFinite() const {
  if (IsMonostate()) {
    return true;
  }
  return WithReal([](const auto &v) { return v.IsFinite(); });
}

bool RealValueImpl::IsNormal() const {
  if (IsMonostate()) {
    return true;
  }
  return WithReal([](const auto &v) { return v.IsNormal(); });
}

int RealValueImpl::Exponent() const {
  if (IsMonostate()) {
    return 0;
  }
  return WithReal([](const auto &v) { return v.Exponent(); });
}

Relation RealValueImpl::Compare(const RealValueImpl &y) const {
  if (IsMonostate()) {
    llvm_unreachable("uncomparable value");
  }
  return WithReal([&](const auto &v) {
    using R = std::decay_t<decltype(v)>;
    return v.Compare(As<R>(y));
  });
}

RealValueImpl RealValueImpl::ABS() const {
  if (IsMonostate()) {
    return RealValueImpl{};
  }
  return WithReal([](const auto &v) { return Wrap(v.ABS()); });
}

RealValueImpl RealValueImpl::Negate() const {
  if (IsMonostate()) {
    return RealValueImpl{};
  }
  return WithReal([](const auto &v) { return Wrap(v.Negate()); });
}

RealValueImpl RealValueImpl::SIGN(const RealValueImpl &x) const {
  if (IsMonostate()) {
    llvm_unreachable("unsupported operation over uninitialized value");
  }
  return WithReal([&](const auto &v) {
    using R = std::decay_t<decltype(v)>;
    return Wrap(v.SIGN(As<R>(x)));
  });
}

RealValueImpl RealValueImpl::SetSign(bool toNegative) const {
  if (IsMonostate()) {
    llvm_unreachable("unsupported operation over uninitialized value");
  }
  return WithReal([&](const auto &v) { return Wrap(v.SetSign(toNegative)); });
}

RealValueImpl RealValueImpl::FlushSubnormalToZero() const {
  if (IsMonostate()) {
    llvm_unreachable("unsupported operation over uninitialized value");
  }
  return WithReal([](const auto &v) { return Wrap(v.FlushSubnormalToZero()); });
}

ValueWithRealFlags<RealValueImpl> RealValueImpl::Add(
    const RealValueImpl &y, Rounding rounding) const {
  if (IsMonostate()) {
    llvm_unreachable("unsupported operation over uninitialized value");
  }
  return WithReal([&](const auto &v) {
    using R = std::decay_t<decltype(v)>;
    return Wrap(v.Add(As<R>(y), rounding));
  });
}

ValueWithRealFlags<RealValueImpl> RealValueImpl::Subtract(
    const RealValueImpl &y, Rounding rounding) const {
  if (IsMonostate()) {
    llvm_unreachable("unsupported operation over uninitialized value");
  }
  return WithReal([&](const auto &v) {
    using R = std::decay_t<decltype(v)>;
    return Wrap(v.Subtract(As<R>(y), rounding));
  });
}

ValueWithRealFlags<RealValueImpl> RealValueImpl::Multiply(
    const RealValueImpl &y, Rounding rounding) const {
  if (IsMonostate()) {
    llvm_unreachable("unsupported operation over uninitialized value");
  }
  return WithReal([&](const auto &v) {
    using R = std::decay_t<decltype(v)>;
    return Wrap(v.Multiply(As<R>(y), rounding));
  });
}

ValueWithRealFlags<RealValueImpl> RealValueImpl::Divide(
    const RealValueImpl &y, Rounding rounding) const {
  if (IsMonostate()) {
    llvm_unreachable("unsupported operation over uninitialized value");
  }
  return WithReal([&](const auto &v) {
    using R = std::decay_t<decltype(v)>;
    return Wrap(v.Divide(As<R>(y), rounding));
  });
}

ValueWithRealFlags<RealValueImpl> RealValueImpl::SQRT(Rounding rounding) const {
  if (IsMonostate()) {
    llvm_unreachable("unsupported operation over uninitialized value");
  }
  return WithReal([&](const auto &v) { return Wrap(v.SQRT(rounding)); });
}

ValueWithRealFlags<RealValueImpl> RealValueImpl::HYPOT(
    const RealValueImpl &y, Rounding rounding) const {
  if (IsMonostate()) {
    llvm_unreachable("unsupported operation over uninitialized value");
  }
  return WithReal([&](const auto &v) {
    using R = std::decay_t<decltype(v)>;
    return Wrap(v.HYPOT(As<R>(y), rounding));
  });
}

ValueWithRealFlags<RealValueImpl> RealValueImpl::MOD(
    const RealValueImpl &y, Rounding rounding) const {
  if (IsMonostate()) {
    llvm_unreachable("unsupported operation over uninitialized value");
  }
  return WithReal([&](const auto &v) {
    using R = std::decay_t<decltype(v)>;
    return Wrap(v.MOD(As<R>(y), rounding));
  });
}

ValueWithRealFlags<RealValueImpl> RealValueImpl::MODULO(
    const RealValueImpl &y, Rounding rounding) const {
  if (IsMonostate()) {
    llvm_unreachable("unsupported operation over uninitialized value");
  }
  return WithReal([&](const auto &v) {
    using R = std::decay_t<decltype(v)>;
    return Wrap(v.MODULO(As<R>(y), rounding));
  });
}

ValueWithRealFlags<RealValueImpl> RealValueImpl::DIM(
    const RealValueImpl &y, Rounding rounding) const {
  if (IsMonostate()) {
    llvm_unreachable("unsupported operation over uninitialized value");
  }
  return WithReal([&](const auto &v) {
    using R = std::decay_t<decltype(v)>;
    return Wrap(v.DIM(As<R>(y), rounding));
  });
}

RealValueImpl RealValueImpl::FRACTION() const {
  if (IsMonostate()) {
    llvm_unreachable("unsupported operation over uninitialized value");
  }
  return WithReal([](const auto &v) { return Wrap(v.FRACTION()); });
}

RealValueImpl RealValueImpl::RRSPACING() const {
  if (IsMonostate()) {
    llvm_unreachable("unsupported operation over uninitialized value");
  }
  return WithReal([](const auto &v) { return Wrap(v.RRSPACING()); });
}

RealValueImpl RealValueImpl::SPACING() const {
  if (IsMonostate()) {
    llvm_unreachable("unsupported operation over uninitialized value");
  }
  return WithReal([](const auto &v) { return Wrap(v.SPACING()); });
}

RealValueImpl RealValueImpl::SET_EXPONENT(std::int64_t e) const {
  if (IsMonostate()) {
    llvm_unreachable("unsupported operation over uninitialized value");
  }
  return WithReal([&](const auto &v) { return Wrap(v.SET_EXPONENT(e)); });
}

ValueWithRealFlags<RealValueImpl> RealValueImpl::NEAREST(bool upward) const {
  if (IsMonostate()) {
    llvm_unreachable("unsupported operation over uninitialized value");
  }
  return WithReal([&](const auto &v) { return Wrap(v.NEAREST(upward)); });
}

ValueWithRealFlags<RealValueImpl> RealValueImpl::ToWholeNumber(
    common::RoundingMode mode) const {
  if (IsMonostate()) {
    llvm_unreachable("unsupported operation over uninitialized value");
  }
  return WithReal([&](const auto &v) { return Wrap(v.ToWholeNumber(mode)); });
}

ValueWithRealFlags<IntegerValue> RealValueImpl::ToInteger(
    common::RoundingMode mode, int toBits) const {
  if (IsMonostate()) {
    return ValueWithRealFlags<IntegerValue>{};
  }
  return WithReal([&](const auto &v) -> ValueWithRealFlags<IntegerValue> {
    auto pick{[&](auto target) -> ValueWithRealFlags<IntegerValue> {
      using W = decltype(target);
      auto r{v.template ToInteger<W>(mode)};
      ValueWithRealFlags<IntegerValue> result;
      result.value = IntegerValueFromFixed(r.value);
      result.flags = r.flags;
      return result;
    }};
    switch (toBits) {
    case 8:
      return pick(Integer<8>{});
    case 16:
      return pick(Integer<16>{});
    case 32:
      return pick(Integer<32>{});
    case 64:
      return pick(Integer<64>{});
    case 128:
      return pick(Integer<128>{});
    default:
      return pick(Integer<64>{});
    }
  });
}

ValueWithRealFlags<RealValueImpl> RealValueImpl::SCALE(
    const IntegerValue &by, Rounding rounding) const {
  if (IsMonostate()) {
    return ValueWithRealFlags<RealValueImpl>{};
  }
  return WithReal([&](const auto &v) -> ValueWithRealFlags<RealValueImpl> {
    return Wrap(v.SCALE(Integer<64>{by.ToInt64()}, rounding));
  });
}

IntegerValue RealValueImpl::EXPONENT() const {
  if (IsMonostate()) {
    return IntegerValue{};
  }
  return WithReal([](const auto &v) -> IntegerValue {
    return IntegerValueFromFixed(v.template EXPONENT<Integer<32>>());
  });
}

ValueWithRealFlags<RealValueImpl> RealValueImpl::KahanSummation(
    const RealValueImpl &y, RealValueImpl &correction,
    Rounding rounding) const {
  if (IsMonostate()) {
    llvm_unreachable("unsupported operation over uninitialized value");
  }
  return WithReal([&](const auto &v) {
    using R = std::decay_t<decltype(v)>;
    R corr{As<R>(correction)};
    auto r{v.KahanSummation(As<R>(y), corr, rounding)};
    correction = Wrap(corr);
    return Wrap(r);
  });
}

ValueWithRealFlags<RealValueImpl> RealValueImpl::FromInteger(
    const IntegerValue &n, int kind, bool isUnsigned, Rounding rounding) {
  if (n.IsMonostate()) {
    return ValueWithRealFlags<RealValueImpl>{};
  }
  return realWithKind(
      kind, [&](auto proto) -> ValueWithRealFlags<RealValueImpl> {
        auto r{FromIntegerValue<decltype(proto)>(n, isUnsigned, rounding)};
        return {Wrap(r.value), r.flags};
      });
}

ValueWithRealFlags<RealValueImpl> RealValueImpl::Convert(
    const RealValueImpl &from, int kind, Rounding rounding) {
  return realWithKind(
      kind, [&](auto proto) -> ValueWithRealFlags<RealValueImpl> {
        using R = decltype(proto);
        if (from.IsMonostate()) {
          return Wrap(R::Convert(R{}, rounding));
        }
        return from.WithReal(
            [&](const auto &v) -> ValueWithRealFlags<RealValueImpl> {
              return Wrap(R::Convert(v, rounding));
            });
      });
}

ValueWithRealFlags<RealValueImpl> RealValueImpl::Read(
    const char *&pp, int kind, Rounding rounding) {
  return realWithKind(
      kind, [&](auto proto) -> ValueWithRealFlags<RealValueImpl> {
        auto r{decltype(proto)::Read(pp, rounding)};
        ValueWithRealFlags<RealValueImpl> result;
        result.value = Wrap(r.value);
        result.flags = r.flags;
        return result;
      });
}

std::string RealValueImpl::DumpHexadecimal() const {
  if (IsMonostate()) {
    llvm_unreachable("unsupported operation over uninitialized value");
  }
  return WithReal([](const auto &v) { return v.DumpHexadecimal(); });
}

llvm::raw_ostream &RealValueImpl::AsFortran(
    llvm::raw_ostream &o, int kind, bool minimal) const {
  if (IsMonostate()) {
    o << "0";
    return o;
  }
  WithReal([&](const auto &v) {
    v.AsFortran(o, kind, minimal);
    return 0;
  });
  return o;
}

} // namespace Fortran::evaluate::value
