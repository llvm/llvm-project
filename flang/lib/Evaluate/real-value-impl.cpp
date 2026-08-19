//===-- lib/Evaluate/real-value-impl.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "real-value-impl.h"
#include "integer-value-impl.h"
#include "flang/Common/idioms.h"
#include "flang/Decimal/decimal.h"
#include "flang/Evaluate/integer-value.h"
#include "flang/Evaluate/real-value.h"
#include "flang/Evaluate/rounding-bits.h"
#include "llvm/Support/raw_ostream.h"
#include <cmath>
#include <cstring>
#include <new>
#include <string>

namespace Fortran::evaluate::value {

RealValueImpl::RealValueImpl(int kind, const Word &w) {
  withWordProto(kind, [&](auto proto) {
    using R = decltype(proto);
    if (w.IsMonostate()) {
      storage_ = R{};
    } else {
      storage_ =
          R{IntegerValueImpl::CoerceUnsigned<typename R::Word>(w.impl())};
    }
  });
}

RealValueImpl::RealValueImpl(int kind, double x) {
  if (x == 0.0) {
    storage_ = std::signbit(x) ? RealValueImpl::NegativeZero(kind).storage_
                               : RealValueImpl::Zero(kind).storage_;
  } else if (std::isnan(x)) {
    storage_ = RealValueImpl::NotANumber(static_cast<KindsEnum>(kind)).storage_;
  } else if (std::isinf(x)) {
    storage_ = RealValueImpl::Infinity(kind, x < 0).storage_;
  } else {
    const bool negative{x < 0};
    int exp{0};
    const double frac{std::frexp(std::fabs(x), &exp)}; // x == +/-frac * 2**exp
    constexpr int fracBits{53}; // exact for any host "double" mantissa
    const auto mantissa{static_cast<std::int64_t>(std::ldexp(frac, fracBits))};
    // Materialize the value in a kind with ample exponent range (IEEE double)
    // first: some target kinds (e.g. REAL(2), a 5-bit-exponent IEEE half) have
    // far too little range to hold the unscaled 53-bit mantissa, and would
    // spuriously overflow to infinity before SCALE() could bring it back down.
    // Convert() then applies the target kind's own IEEE rounding/overflow
    // semantics for the final narrowing (or widening).
    constexpr int wideKind{8};
    RealValueImpl magnitude{RealValueImpl::FromInteger(
        wideKind, IntegerValue{KindsEnum::Kind8, mantissa})
            .value};
    magnitude =
        magnitude.SCALE(IntegerValue{KindsEnum::Kind4, exp - fracBits}).value;
    if (negative) {
      magnitude = magnitude.SetSign(true);
    }
    storage_ = (kind == wideKind)
        ? magnitude.storage_
        : RealValueImpl::Convert(kind, magnitude).value.storage_;
  }
}

RealValueImpl RealValueImpl::Zero(int kind) {
  RealValueImpl result;
  withWordProto(kind, [&](auto proto) { result.storage_ = decltype(proto){}; });
  return result;
}

RealValueImpl RealValueImpl::FromRawBytes(
    int kind, const void *raw, std::size_t expectedSize) {
  return RealValueImpl{kind,
      IntegerValue::FromRawBytes(
          static_cast<KindsEnum>(kind), raw, expectedSize)};
}

void RealValueImpl::print(llvm::raw_ostream &os) const {
  AsFortran(os, static_cast<int>(kind()));
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void RealValueImpl::dump() const {
  print(llvm::errs());
  llvm::errs() << '\n';
}
#endif

KindsEnum RealValueImpl::kind() const {
  if (IsMonostate()) {
    llvm_unreachable("uninitialized value has not a defined kind");
  }

  return withWord([](const auto &v) -> KindsEnum {
    using R = std::decay_t<decltype(v)>;
    if constexpr (std::is_same_v<R, R3>) {
      return KindsEnum::Kind3;
    }
    return static_cast<KindsEnum>(R::bits / 8);
  });
}

int RealValueImpl::bits() const {
  if (IsMonostate()) {
    return 0;
  }

  return withWord(
      [](const auto &v) -> int { return std::decay_t<decltype(v)>::bits; });
}

bool RealValueImpl::IsZero() const {
  if (IsMonostate()) {
    return true;
  }
  return withWord([](const auto &v) { return v.IsZero(); });
}

bool RealValueImpl::operator==(const RealValueImpl &y) const {
  return withWord([&y](const auto &v1) -> bool {
    return y.withWord([&v1](const auto &v2) -> bool {
      if constexpr (std::is_same_v<std::decay_t<decltype(v1)>,
                        std::decay_t<decltype(v2)>>) {
        return v1 == v2;
      }
      llvm_unreachable("Uncomparable reals");
    });
  });
}

int RealValueImpl::DIGITS(int kind) {
  return withWordProto(kind, [](auto p) { return decltype(p)::DIGITS; });
}

int RealValueImpl::PRECISION(int kind) {
  return withWordProto(kind, [](auto p) { return decltype(p)::PRECISION; });
}

int RealValueImpl::RANGE(int kind) {
  return withWordProto(kind, [](auto p) { return decltype(p)::RANGE; });
}

int RealValueImpl::MAXEXPONENT(int kind) {
  return withWordProto(kind, [](auto p) { return decltype(p)::MAXEXPONENT; });
}

int RealValueImpl::MINEXPONENT(int kind) {
  return withWordProto(kind, [](auto p) { return decltype(p)::MINEXPONENT; });
}

RealValueImpl RealValueImpl::HUGE(int kind) {
  return withWordProto(
      kind, [](auto p) { return FromWord(decltype(p)::HUGE()); });
}

RealValueImpl RealValueImpl::EPSILON(int kind) {
  return withWordProto(
      kind, [](auto p) { return FromWord(decltype(p)::EPSILON()); });
}

RealValueImpl RealValueImpl::TINY(int kind) {
  return withWordProto(
      kind, [](auto p) { return FromWord(decltype(p)::TINY()); });
}

RealValueImpl RealValueImpl::NotANumber(KindsEnum kind) {
  return withWordProto(static_cast<int>(kind),
      [](auto p) { return FromWord(decltype(p)::NotANumber()); });
}

RealValueImpl RealValueImpl::SignalingNaN(int kind) {
  return withWordProto(
      kind, [](auto p) { return FromWord(decltype(p)::SignalingNaN()); });
}

RealValueImpl RealValueImpl::Infinity(int kind, bool negative) {
  return withWordProto(kind,
      [negative](auto p) { return FromWord(decltype(p)::Infinity(negative)); });
}

RealValueImpl RealValueImpl::NegativeZero(int kind) {
  return withWordProto(
      kind, [](auto p) { return FromWord(decltype(p)::NegativeZero()); });
}

bool RealValueImpl::IsNegative() const {
  if (IsMonostate()) {
    return false;
  }
  return withWord([](const auto &v) { return v.IsNegative(); });
}

bool RealValueImpl::IsNotANumber() const {
  if (IsMonostate()) {
    return false;
  }
  return withWord([](const auto &v) { return v.IsNotANumber(); });
}

bool RealValueImpl::IsSignalingNaN() const {
  if (IsMonostate()) {
    return false;
  }
  return withWord([](const auto &v) { return v.IsSignalingNaN(); });
}

bool RealValueImpl::IsInfinite() const {
  if (IsMonostate()) {
    return false;
  }
  return withWord([](const auto &v) { return v.IsInfinite(); });
}

bool RealValueImpl::IsFinite() const {
  if (IsMonostate()) {
    return true;
  }
  return withWord([](const auto &v) { return v.IsFinite(); });
}

bool RealValueImpl::IsNormal() const {
  if (IsMonostate()) {
    return true;
  }
  return withWord([](const auto &v) { return v.IsNormal(); });
}

int RealValueImpl::Exponent() const {
  if (IsMonostate()) {
    return 0;
  }
  return withWord([](const auto &v) { return v.Exponent(); });
}

void RealValueImpl::StoreRawBytes(
    void *dst, size_t expectedSize, bool *changed) const {
  CHECK(bytesStored() == expectedSize);
  withWord([=](const auto &v) {
    auto data{v.RawBits()};
    CHECK(sizeof(data) == expectedSize);
    if (std::memcmp(dst, &data, sizeof(data))) {
      std::memcpy(dst, &data, sizeof(data));
      if (changed)
        *changed = true;
    }
  });
}

IntegerValue RealValueImpl::RawBits() const {
  if (IsMonostate()) {
    return {};
  }

  return withWord([](const auto &v) {
    IntegerValue result;
    result.impl() = IntegerValueImpl::FromWord(v.RawBits());
    return result;
  });
}

Relation RealValueImpl::Compare(const RealValueImpl &y) const {
  if (IsMonostate()) {
    llvm_unreachable("uncomparable value");
  }
  return withWord([&](const auto &v) {
    using R = std::decay_t<decltype(v)>;
    return v.Compare(AsWord<R>(y));
  });
}

RealValueImpl RealValueImpl::ABS() const {
  if (IsMonostate()) {
    return RealValueImpl{};
  }
  return withWord([](const auto &v) { return FromWord(v.ABS()); });
}

RealValueImpl RealValueImpl::Negate() const {
  if (IsMonostate()) {
    return RealValueImpl{};
  }
  return withWord([](const auto &v) { return FromWord(v.Negate()); });
}

RealValueImpl RealValueImpl::SIGN(const RealValueImpl &x) const {
  if (IsMonostate()) {
    llvm_unreachable("unsupported operation over uninitialized value");
  }
  return withWord([&](const auto &v) {
    using R = std::decay_t<decltype(v)>;
    return FromWord(v.SIGN(AsWord<R>(x)));
  });
}

RealValueImpl RealValueImpl::SetSign(bool toNegative) const {
  if (IsMonostate()) {
    llvm_unreachable("unsupported operation over uninitialized value");
  }
  return withWord(
      [&](const auto &v) { return FromWord(v.SetSign(toNegative)); });
}

RealValueImpl RealValueImpl::FlushSubnormalToZero() const {
  if (IsMonostate()) {
    llvm_unreachable("unsupported operation over uninitialized value");
  }
  return withWord(
      [](const auto &v) { return FromWord(v.FlushSubnormalToZero()); });
}

ValueWithRealFlags<RealValueImpl> RealValueImpl::Add(
    const RealValueImpl &y, Rounding rounding) const {
  if (IsMonostate()) {
    llvm_unreachable("unsupported operation over uninitialized value");
  }
  return withWord([&](const auto &v) {
    using R = std::decay_t<decltype(v)>;
    return FromWord(v.Add(AsWord<R>(y), rounding));
  });
}

ValueWithRealFlags<RealValueImpl> RealValueImpl::Subtract(
    const RealValueImpl &y, Rounding rounding) const {
  if (IsMonostate()) {
    llvm_unreachable("unsupported operation over uninitialized value");
  }
  return withWord([&](const auto &v) {
    using R = std::decay_t<decltype(v)>;
    return FromWord(v.Subtract(AsWord<R>(y), rounding));
  });
}

ValueWithRealFlags<RealValueImpl> RealValueImpl::Multiply(
    const RealValueImpl &y, Rounding rounding) const {
  if (IsMonostate()) {
    llvm_unreachable("unsupported operation over uninitialized value");
  }
  return withWord([&](const auto &v) {
    using R = std::decay_t<decltype(v)>;
    return FromWord(v.Multiply(AsWord<R>(y), rounding));
  });
}

ValueWithRealFlags<RealValueImpl> RealValueImpl::Divide(
    const RealValueImpl &y, Rounding rounding) const {
  if (IsMonostate()) {
    llvm_unreachable("unsupported operation over uninitialized value");
  }
  return withWord([&](const auto &v) {
    using R = std::decay_t<decltype(v)>;
    return FromWord(v.Divide(AsWord<R>(y), rounding));
  });
}

ValueWithRealFlags<RealValueImpl> RealValueImpl::SQRT(Rounding rounding) const {
  if (IsMonostate()) {
    llvm_unreachable("unsupported operation over uninitialized value");
  }
  return withWord([&](const auto &v) { return FromWord(v.SQRT(rounding)); });
}

ValueWithRealFlags<RealValueImpl> RealValueImpl::HYPOT(
    const RealValueImpl &y, Rounding rounding) const {
  if (IsMonostate()) {
    llvm_unreachable("unsupported operation over uninitialized value");
  }
  return withWord([&](const auto &v) {
    using R = std::decay_t<decltype(v)>;
    return FromWord(v.HYPOT(AsWord<R>(y), rounding));
  });
}

ValueWithRealFlags<RealValueImpl> RealValueImpl::MOD(
    const RealValueImpl &y, Rounding rounding) const {
  if (IsMonostate()) {
    llvm_unreachable("unsupported operation over uninitialized value");
  }
  return withWord([&](const auto &v) {
    using R = std::decay_t<decltype(v)>;
    return FromWord(v.MOD(AsWord<R>(y), rounding));
  });
}

ValueWithRealFlags<RealValueImpl> RealValueImpl::MODULO(
    const RealValueImpl &y, Rounding rounding) const {
  if (IsMonostate()) {
    llvm_unreachable("unsupported operation over uninitialized value");
  }
  return withWord([&](const auto &v) {
    using R = std::decay_t<decltype(v)>;
    return FromWord(v.MODULO(AsWord<R>(y), rounding));
  });
}

ValueWithRealFlags<RealValueImpl> RealValueImpl::DIM(
    const RealValueImpl &y, Rounding rounding) const {
  if (IsMonostate()) {
    llvm_unreachable("unsupported operation over uninitialized value");
  }
  return withWord([&](const auto &v) {
    using R = std::decay_t<decltype(v)>;
    return FromWord(v.DIM(AsWord<R>(y), rounding));
  });
}

RealValueImpl RealValueImpl::FRACTION() const {
  if (IsMonostate()) {
    llvm_unreachable("unsupported operation over uninitialized value");
  }
  return withWord([](const auto &v) { return FromWord(v.FRACTION()); });
}

RealValueImpl RealValueImpl::RRSPACING() const {
  if (IsMonostate()) {
    llvm_unreachable("unsupported operation over uninitialized value");
  }
  return withWord([](const auto &v) { return FromWord(v.RRSPACING()); });
}

RealValueImpl RealValueImpl::SPACING() const {
  if (IsMonostate()) {
    llvm_unreachable("unsupported operation over uninitialized value");
  }
  return withWord([](const auto &v) { return FromWord(v.SPACING()); });
}

RealValueImpl RealValueImpl::SET_EXPONENT(std::int64_t e) const {
  if (IsMonostate()) {
    llvm_unreachable("unsupported operation over uninitialized value");
  }
  return withWord([&](const auto &v) { return FromWord(v.SET_EXPONENT(e)); });
}

ValueWithRealFlags<RealValueImpl> RealValueImpl::NEAREST(bool upward) const {
  if (IsMonostate()) {
    llvm_unreachable("unsupported operation over uninitialized value");
  }
  return withWord([&](const auto &v) { return FromWord(v.NEAREST(upward)); });
}

ValueWithRealFlags<RealValueImpl> RealValueImpl::ToWholeNumber(
    common::RoundingMode mode) const {
  if (IsMonostate()) {
    llvm_unreachable("unsupported operation over uninitialized value");
  }
  return withWord(
      [&](const auto &v) { return FromWord(v.ToWholeNumber(mode)); });
}

ValueWithRealFlags<IntegerValue> RealValueImpl::ToInteger(
    common::RoundingMode mode, int toBits) const {
  if (IsMonostate()) {
    llvm_unreachable("unsupported operation over uninitialized value");
  }
  return withWord([&](const auto &v) -> ValueWithRealFlags<IntegerValue> {
    auto pick{[&](auto target) -> ValueWithRealFlags<IntegerValue> {
      using W = decltype(target);
      auto r{v.template ToInteger<W>(mode)};
      ValueWithRealFlags<IntegerValue> result;
      result.value.impl() = IntegerValueImpl::FromWord(r.value);
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
    llvm_unreachable("unsupported operation over uninitialized value");
  }
  return withWord([&](const auto &v) -> ValueWithRealFlags<RealValueImpl> {
    return FromWord(v.SCALE(Integer<64>{by.ToInt64()}, rounding));
  });
}

ValueWithRealFlags<RealValueImpl> RealValueImpl::KahanSummation(
    const RealValueImpl &y, RealValueImpl &correction,
    Rounding rounding) const {
  if (IsMonostate()) {
    llvm_unreachable("unsupported operation over uninitialized value");
  }
  return withWord([&](const auto &v) {
    using R = std::decay_t<decltype(v)>;
    R corr{AsWord<R>(correction)};
    auto r{v.KahanSummation(AsWord<R>(y), corr, rounding)};
    correction = FromWord(corr);
    return FromWord(r);
  });
}

IntegerValue RealValueImpl::EXPONENT() const {
  if (IsMonostate()) {
    llvm_unreachable("unsupported operation over uninitialized value");
  }
  return withWord([](const auto &v) -> IntegerValue {
    IntegerValue result;
    result.impl() =
        IntegerValueImpl::FromWord(v.template EXPONENT<Integer<32>>());
    return result;
  });
}

ValueWithRealFlags<RealValueImpl> RealValueImpl::FromInteger(
    int kind, const IntegerValue &n, bool isUnsigned, Rounding rounding) {
  if (n.IsMonostate()) {
    return ValueWithRealFlags<RealValueImpl>{};
  }
  return withWordProto(
      kind, [&](auto proto) -> ValueWithRealFlags<RealValueImpl> {
        using R = std::decay_t<decltype(proto)>;
        auto r{n.impl().withWord([&](const auto &concrete) {
          return R::FromInteger(concrete, isUnsigned, rounding);
        })};
        return {FromWord(r.value), r.flags};
      });
}

ValueWithRealFlags<RealValueImpl> RealValueImpl::Convert(
    int kind, const RealValueImpl &from, Rounding rounding) {
  return withWordProto(
      kind, [&](auto proto) -> ValueWithRealFlags<RealValueImpl> {
        using R = decltype(proto);
        if (from.IsMonostate()) {
          return FromWord(R::Convert(R{}, rounding));
        }
        return from.withWord(
            [&](const auto &v) -> ValueWithRealFlags<RealValueImpl> {
              return FromWord(R::Convert(v, rounding));
            });
      });
}

ValueWithRealFlags<RealValueImpl> RealValueImpl::Read(
    int kind, const char *&pp, Rounding rounding) {
  return withWordProto(
      kind, [&](auto proto) -> ValueWithRealFlags<RealValueImpl> {
        auto r{decltype(proto)::Read(pp, rounding)};
        ValueWithRealFlags<RealValueImpl> result;
        result.value = FromWord(r.value);
        result.flags = r.flags;
        return result;
      });
}

std::string RealValueImpl::DumpHexadecimal() const {
  if (IsMonostate()) {
    llvm_unreachable("unsupported operation over uninitialized value");
  }
  return withWord([](const auto &v) { return v.DumpHexadecimal(); });
}

llvm::raw_ostream &RealValueImpl::AsFortran(
    llvm::raw_ostream &o, int kind, bool minimal) const {
  if (IsMonostate()) {
    o << "0";
    return o;
  }
  withWord([&](const auto &v) {
    v.AsFortran(o, kind, minimal);
    return 0;
  });
  return o;
}

} // namespace Fortran::evaluate::value
