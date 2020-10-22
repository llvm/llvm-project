//===-- lib/Evaluate/real.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Evaluate/real.h"
#include "int-power.h"
#include "flang/Common/idioms.h"
#include "flang/Decimal/decimal.h"
#include "flang/Parser/characters.h"
#include "llvm/Support/raw_ostream.h"
#include <limits>

namespace Fortran::evaluate::value {

template <typename W, int P> Relation Real<W, P>::Compare(const Real &y) const {
  if (IsNotANumber() || y.IsNotANumber()) { // NaN vs x, x vs NaN
    return Relation::Unordered;
  } else if (IsInfinite()) {
    if (y.IsInfinite()) {
      if (IsNegative()) { // -Inf vs +/-Inf
        return y.IsNegative() ? Relation::Equal : Relation::Less;
      } else { // +Inf vs +/-Inf
        return y.IsNegative() ? Relation::Greater : Relation::Equal;
      }
    } else { // +/-Inf vs finite
      return IsNegative() ? Relation::Less : Relation::Greater;
    }
  } else if (y.IsInfinite()) { // finite vs +/-Inf
    return y.IsNegative() ? Relation::Greater : Relation::Less;
  } else { // two finite numbers
    bool isNegative{IsNegative()};
    if (isNegative != y.IsNegative()) {
      if (word_.IOR(y.word_).IBCLR(bits - 1).IsZero()) {
        return Relation::Equal; // +/-0.0 == -/+0.0
      } else {
        return isNegative ? Relation::Less : Relation::Greater;
      }
    } else {
      // same sign
      Ordering order{evaluate::Compare(Exponent(), y.Exponent())};
      if (order == Ordering::Equal) {
        order = GetSignificand().CompareUnsigned(y.GetSignificand());
      }
      if (isNegative) {
        order = Reverse(order);
      }
      return RelationFromOrdering(order);
    }
  }
}

template <typename W, int P>
ValueWithRealFlags<Real<W, P>> Real<W, P>::Add(
    const Real &y, Rounding rounding) const {
  ValueWithRealFlags<Real> result;
  if (IsNotANumber() || y.IsNotANumber()) {
    result.value = NotANumber(); // NaN + x -> NaN
    if (IsSignalingNaN() || y.IsSignalingNaN()) {
      result.flags.set(RealFlag::InvalidArgument);
    }
    return result;
  }
  bool isNegative{IsNegative()};
  bool yIsNegative{y.IsNegative()};
  if (IsInfinite()) {
    if (y.IsInfinite()) {
      if (isNegative == yIsNegative) {
        result.value = *this; // +/-Inf + +/-Inf -> +/-Inf
      } else {
        result.value = NotANumber(); // +/-Inf + -/+Inf -> NaN
        result.flags.set(RealFlag::InvalidArgument);
      }
    } else {
      result.value = *this; // +/-Inf + x -> +/-Inf
    }
    return result;
  }
  if (y.IsInfinite()) {
    result.value = y; // x + +/-Inf -> +/-Inf
    return result;
  }
  int exponent{Exponent()};
  int yExponent{y.Exponent()};
  if (exponent < yExponent) {
    // y is larger in magnitude; simplify by reversing operands
    return y.Add(*this, rounding);
  }
  if (exponent == yExponent && isNegative != yIsNegative) {
    Ordering order{GetSignificand().CompareUnsigned(y.GetSignificand())};
    if (order == Ordering::Less) {
      // Same exponent, opposite signs, and y is larger in magnitude
      return y.Add(*this, rounding);
    }
    if (order == Ordering::Equal) {
      // x + (-x) -> +0.0 unless rounding is directed downwards
      if (rounding.mode == common::RoundingMode::Down) {
        result.value.word_ = result.value.word_.IBSET(bits - 1); // -0.0
      }
      return result;
    }
  }
  // Our exponent is greater than y's, or the exponents match and y is not
  // of the opposite sign and greater magnitude.  So (x+y) will have the
  // same sign as x.
  Fraction fraction{GetFraction()};
  Fraction yFraction{y.GetFraction()};
  int rshift = exponent - yExponent;
  if (exponent > 0 && yExponent == 0) {
    --rshift; // correct overshift when only y is subnormal
  }
  RoundingBits roundingBits{yFraction, rshift};
  yFraction = yFraction.SHIFTR(rshift);
  bool carry{false};
  if (isNegative != yIsNegative) {
    // Opposite signs: subtract via addition of two's complement of y and
    // the rounding bits.
    yFraction = yFraction.NOT();
    carry = roundingBits.Negate();
  }
  auto sum{fraction.AddUnsigned(yFraction, carry)};
  fraction = sum.value;
  if (isNegative == yIsNegative && sum.carry) {
    roundingBits.ShiftRight(sum.value.BTEST(0));
    fraction = fraction.SHIFTR(1).IBSET(fraction.bits - 1);
    ++exponent;
  }
  NormalizeAndRound(
      result, isNegative, exponent, fraction, rounding, roundingBits);
  return result;
}

template <typename W, int P>
ValueWithRealFlags<Real<W, P>> Real<W, P>::Multiply(
    const Real &y, Rounding rounding) const {
  ValueWithRealFlags<Real> result;
  if (IsNotANumber() || y.IsNotANumber()) {
    result.value = NotANumber(); // NaN * x -> NaN
    if (IsSignalingNaN() || y.IsSignalingNaN()) {
      result.flags.set(RealFlag::InvalidArgument);
    }
  } else {
    bool isNegative{IsNegative() != y.IsNegative()};
    if (IsInfinite() || y.IsInfinite()) {
      if (IsZero() || y.IsZero()) {
        result.value = NotANumber(); // 0 * Inf -> NaN
        result.flags.set(RealFlag::InvalidArgument);
      } else {
        result.value = Infinity(isNegative);
      }
    } else {
      auto product{GetFraction().MultiplyUnsigned(y.GetFraction())};
      std::int64_t exponent{CombineExponents(y, false)};
      if (exponent < 1) {
        int rshift = 1 - exponent;
        exponent = 1;
        bool sticky{false};
        if (rshift >= product.upper.bits + product.lower.bits) {
          sticky = !product.lower.IsZero() || !product.upper.IsZero();
        } else if (rshift >= product.lower.bits) {
          sticky = !product.lower.IsZero() ||
              !product.upper
                   .IAND(product.upper.MASKR(rshift - product.lower.bits))
                   .IsZero();
        } else {
          sticky = !product.lower.IAND(product.lower.MASKR(rshift)).IsZero();
        }
        product.lower = product.lower.SHIFTRWithFill(product.upper, rshift);
        product.upper = product.upper.SHIFTR(rshift);
        if (sticky) {
          product.lower = product.lower.IBSET(0);
        }
      }
      int leadz{product.upper.LEADZ()};
      if (leadz >= product.upper.bits) {
        leadz += product.lower.LEADZ();
      }
      int lshift{leadz};
      if (lshift > exponent - 1) {
        lshift = exponent - 1;
      }
      exponent -= lshift;
      product.upper = product.upper.SHIFTLWithFill(product.lower, lshift);
      product.lower = product.lower.SHIFTL(lshift);
      RoundingBits roundingBits{product.lower, product.lower.bits};
      NormalizeAndRound(result, isNegative, exponent, product.upper, rounding,
          roundingBits, true /*multiply*/);
    }
  }
  return result;
}

template <typename W, int P>
ValueWithRealFlags<Real<W, P>> Real<W, P>::Divide(
    const Real &y, Rounding rounding) const {
  ValueWithRealFlags<Real> result;
  if (IsNotANumber() || y.IsNotANumber()) {
    result.value = NotANumber(); // NaN / x -> NaN, x / NaN -> NaN
    if (IsSignalingNaN() || y.IsSignalingNaN()) {
      result.flags.set(RealFlag::InvalidArgument);
    }
  } else {
    bool isNegative{IsNegative() != y.IsNegative()};
    if (IsInfinite()) {
      if (y.IsInfinite()) {
        result.value = NotANumber(); // Inf/Inf -> NaN
        result.flags.set(RealFlag::InvalidArgument);
      } else { // Inf/x -> Inf,  Inf/0 -> Inf
        result.value = Infinity(isNegative);
      }
    } else if (y.IsZero()) {
      if (IsZero()) { // 0/0 -> NaN
        result.value = NotANumber();
        result.flags.set(RealFlag::InvalidArgument);
      } else { // x/0 -> Inf, Inf/0 -> Inf
        result.value = Infinity(isNegative);
        result.flags.set(RealFlag::DivideByZero);
      }
    } else if (IsZero() || y.IsInfinite()) { // 0/x, x/Inf -> 0
      if (isNegative) {
        result.value.word_ = result.value.word_.IBSET(bits - 1);
      }
    } else {
      // dividend and divisor are both finite and nonzero numbers
      Fraction top{GetFraction()}, divisor{y.GetFraction()};
      std::int64_t exponent{CombineExponents(y, true)};
      Fraction quotient;
      bool msb{false};
      if (!top.BTEST(top.bits - 1) || !divisor.BTEST(divisor.bits - 1)) {
        // One or two subnormals
        int topLshift{top.LEADZ()};
        top = top.SHIFTL(topLshift);
        int divisorLshift{divisor.LEADZ()};
        divisor = divisor.SHIFTL(divisorLshift);
        exponent += divisorLshift - topLshift;
      }
      for (int j{1}; j <= quotient.bits; ++j) {
        if (NextQuotientBit(top, msb, divisor)) {
          quotient = quotient.IBSET(quotient.bits - j);
        }
      }
      bool guard{NextQuotientBit(top, msb, divisor)};
      bool round{NextQuotientBit(top, msb, divisor)};
      bool sticky{msb || !top.IsZero()};
      RoundingBits roundingBits{guard, round, sticky};
      if (exponent < 1) {
        std::int64_t rshift{1 - exponent};
        for (; rshift > 0; --rshift) {
          roundingBits.ShiftRight(quotient.BTEST(0));
          quotient = quotient.SHIFTR(1);
        }
        exponent = 1;
      }
      NormalizeAndRound(
          result, isNegative, exponent, quotient, rounding, roundingBits);
    }
  }
  return result;
}

template <typename W, int P>
ValueWithRealFlags<Real<W, P>> Real<W, P>::ToWholeNumber(
    common::RoundingMode mode) const {
  ValueWithRealFlags<Real> result{*this};
  if (IsNotANumber()) {
    result.flags.set(RealFlag::InvalidArgument);
    result.value = NotANumber();
  } else if (IsInfinite()) {
    result.flags.set(RealFlag::Overflow);
  } else {
    constexpr int noClipExponent{exponentBias + binaryPrecision - 1};
    if (Exponent() < noClipExponent) {
      Real adjust; // ABS(EPSILON(adjust)) == 0.5
      adjust.Normalize(IsSignBitSet(), noClipExponent, Fraction::MASKL(1));
      // Compute ival=(*this + adjust), losing any fractional bits; keep flags
      result = Add(adjust, Rounding{mode});
      result.flags.reset(RealFlag::Inexact); // result *is* exact
      // Return (ival-adjust) with original sign in case we've generated a zero.
      result.value =
          result.value.Subtract(adjust, Rounding{common::RoundingMode::ToZero})
              .value.SIGN(*this);
    }
  }
  return result;
}

template <typename W, int P>
RealFlags Real<W, P>::Normalize(bool negative, int exponent,
    const Fraction &fraction, Rounding rounding, RoundingBits *roundingBits) {
  int lshift{fraction.LEADZ()};
  if (lshift == fraction.bits /* fraction is zero */ &&
      (!roundingBits || roundingBits->empty())) {
    // No fraction, no rounding bits -> +/-0.0
    exponent = lshift = 0;
  } else if (lshift < exponent) {
    exponent -= lshift;
  } else if (exponent > 0) {
    lshift = exponent - 1;
    exponent = 0;
  } else if (lshift == 0) {
    exponent = 1;
  } else {
    lshift = 0;
  }
  if (exponent >= maxExponent) {
    // Infinity or overflow
    if (rounding.mode == common::RoundingMode::TiesToEven ||
        rounding.mode == common::RoundingMode::TiesAwayFromZero ||
        (rounding.mode == common::RoundingMode::Up && !negative) ||
        (rounding.mode == common::RoundingMode::Down && negative)) {
      word_ = Word{maxExponent}.SHIFTL(significandBits); // Inf
    } else {
      // directed rounding: round to largest finite value rather than infinity
      // (x86 does this, not sure whether it's standard behavior)
      word_ = Word{word_.MASKR(word_.bits - 1)}.IBCLR(significandBits);
    }
    if (negative) {
      word_ = word_.IBSET(bits - 1);
    }
    RealFlags flags{RealFlag::Overflow};
    if (!fraction.IsZero()) {
      flags.set(RealFlag::Inexact);
    }
    return flags;
  }
  word_ = Word::ConvertUnsigned(fraction).value;
  if (lshift > 0) {
    word_ = word_.SHIFTL(lshift);
    if (roundingBits) {
      for (; lshift > 0; --lshift) {
        if (roundingBits->ShiftLeft()) {
          word_ = word_.IBSET(lshift - 1);
        }
      }
    }
  }
  if constexpr (isImplicitMSB) {
    word_ = word_.IBCLR(significandBits);
  }
  word_ = word_.IOR(Word{exponent}.SHIFTL(significandBits));
  if (negative) {
    word_ = word_.IBSET(bits - 1);
  }
  return {};
}

template <typename W, int P>
RealFlags Real<W, P>::Round(
    Rounding rounding, const RoundingBits &bits, bool multiply) {
  int origExponent{Exponent()};
  RealFlags flags;
  bool inexact{!bits.empty()};
  if (inexact) {
    flags.set(RealFlag::Inexact);
  }
  if (origExponent < maxExponent &&
      bits.MustRound(rounding, IsNegative(), word_.BTEST(0) /* is odd */)) {
    typename Fraction::ValueWithCarry sum{
        GetFraction().AddUnsigned(Fraction{}, true)};
    int newExponent{origExponent};
    if (sum.carry) {
      // The fraction was all ones before rounding; sum.value is now zero
      sum.value = sum.value.IBSET(binaryPrecision - 1);
      if (++newExponent >= maxExponent) {
        flags.set(RealFlag::Overflow); // rounded away to an infinity
      }
    }
    flags |= Normalize(IsNegative(), newExponent, sum.value);
  }
  if (inexact && origExponent == 0) {
    // inexact subnormal input: signal Underflow unless in an x86-specific
    // edge case
    if (rounding.x86CompatibleBehavior && Exponent() != 0 && multiply &&
        bits.sticky() &&
        (bits.guard() ||
            (rounding.mode != common::RoundingMode::Up &&
                rounding.mode != common::RoundingMode::Down))) {
      // x86 edge case in which Underflow fails to signal when a subnormal
      // inexact multiplication product rounds to a normal result when
      // the guard bit is set or we're not using directed rounding
    } else {
      flags.set(RealFlag::Underflow);
    }
  }
  return flags;
}

template <typename W, int P>
void Real<W, P>::NormalizeAndRound(ValueWithRealFlags<Real> &result,
    bool isNegative, int exponent, const Fraction &fraction, Rounding rounding,
    RoundingBits roundingBits, bool multiply) {
  result.flags |= result.value.Normalize(
      isNegative, exponent, fraction, rounding, &roundingBits);
  result.flags |= result.value.Round(rounding, roundingBits, multiply);
}

inline enum decimal::FortranRounding MapRoundingMode(
    common::RoundingMode rounding) {
  switch (rounding) {
  case common::RoundingMode::TiesToEven:
    break;
  case common::RoundingMode::ToZero:
    return decimal::RoundToZero;
  case common::RoundingMode::Down:
    return decimal::RoundDown;
  case common::RoundingMode::Up:
    return decimal::RoundUp;
  case common::RoundingMode::TiesAwayFromZero:
    return decimal::RoundCompatible;
  }
  return decimal::RoundNearest; // dodge gcc warning about lack of result
}

inline RealFlags MapFlags(decimal::ConversionResultFlags flags) {
  RealFlags result;
  if (flags & decimal::Overflow) {
    result.set(RealFlag::Overflow);
  }
  if (flags & decimal::Inexact) {
    result.set(RealFlag::Inexact);
  }
  if (flags & decimal::Invalid) {
    result.set(RealFlag::InvalidArgument);
  }
  return result;
}

template <typename W, int P>
ValueWithRealFlags<Real<W, P>> Real<W, P>::Read(
    const char *&p, Rounding rounding) {
  auto converted{
      decimal::ConvertToBinary<P>(p, MapRoundingMode(rounding.mode))};
  const auto *value{reinterpret_cast<Real<W, P> *>(&converted.binary)};
  return {*value, MapFlags(converted.flags)};
}

template <typename W, int P> std::string Real<W, P>::DumpHexadecimal() const {
  if (IsNotANumber()) {
    return "NaN 0x"s + word_.Hexadecimal();
  } else if (IsNegative()) {
    return "-"s + Negate().DumpHexadecimal();
  } else if (IsInfinite()) {
    return "Inf"s;
  } else if (IsZero()) {
    return "0.0"s;
  } else {
    Fraction frac{GetFraction()};
    std::string result{"0x"};
    char intPart = '0' + frac.BTEST(frac.bits - 1);
    result += intPart;
    result += '.';
    int trailz{frac.TRAILZ()};
    if (trailz >= frac.bits - 1) {
      result += '0';
    } else {
      int remainingBits{frac.bits - 1 - trailz};
      int wholeNybbles{remainingBits / 4};
      int lostBits{remainingBits - 4 * wholeNybbles};
      if (wholeNybbles > 0) {
        std::string fracHex{frac.SHIFTR(trailz + lostBits)
                                .IAND(frac.MASKR(4 * wholeNybbles))
                                .Hexadecimal()};
        std::size_t field = wholeNybbles;
        if (fracHex.size() < field) {
          result += std::string(field - fracHex.size(), '0');
        }
        result += fracHex;
      }
      if (lostBits > 0) {
        result += frac.SHIFTR(trailz)
                      .IAND(frac.MASKR(lostBits))
                      .SHIFTL(4 - lostBits)
                      .Hexadecimal();
      }
    }
    result += 'p';
    int exponent = Exponent() - exponentBias;
    result += Integer<32>{exponent}.SignedDecimal();
    return result;
  }
}

template <typename W, int P>
llvm::raw_ostream &Real<W, P>::AsFortran(
    llvm::raw_ostream &o, int kind, bool minimal) const {
  if (IsNotANumber()) {
    o << "(0._" << kind << "/0.)";
  } else if (IsInfinite()) {
    if (IsNegative()) {
      o << "(-1._" << kind << "/0.)";
    } else {
      o << "(1._" << kind << "/0.)";
    }
  } else {
    using B = decimal::BinaryFloatingPointNumber<P>;
    B value{word_.template ToUInt<typename B::RawType>()};
    char buffer[common::MaxDecimalConversionDigits(P) +
        EXTRA_DECIMAL_CONVERSION_SPACE];
    decimal::DecimalConversionFlags flags{}; // default: exact representation
    if (minimal) {
      flags = decimal::Minimize;
    }
    auto result{decimal::ConvertToDecimal<P>(buffer, sizeof buffer, flags,
        static_cast<int>(sizeof buffer), decimal::RoundNearest, value)};
    const char *p{result.str};
    if (DEREF(p) == '-' || *p == '+') {
      o << *p++;
    }
    int expo{result.decimalExponent};
    if (*p != '0') {
      --expo;
    }
    o << *p << '.' << (p + 1);
    if (expo != 0) {
      o << 'e' << expo;
    }
    o << '_' << kind;
  }
  return o;
}

template class Real<Integer<16>, 11>;
template class Real<Integer<16>, 8>;
template class Real<Integer<32>, 24>;
template class Real<Integer<64>, 53>;
template class Real<Integer<80>, 64>;
template class Real<Integer<128>, 113>;
} // namespace Fortran::evaluate::value
