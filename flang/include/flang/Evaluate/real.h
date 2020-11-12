//===-- include/flang/Evaluate/real.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_EVALUATE_REAL_H_
#define FORTRAN_EVALUATE_REAL_H_

#include "formatting.h"
#include "integer.h"
#include "rounding-bits.h"
#include "flang/Common/real.h"
#include "flang/Evaluate/common.h"
#include <cinttypes>
#include <limits>
#include <string>

// Some environments, viz. clang on Darwin, allow the macro HUGE
// to leak out of <math.h> even when it is never directly included.
#undef HUGE

namespace llvm {
class raw_ostream;
}
namespace Fortran::evaluate::value {

// LOG10(2.)*1E12
static constexpr std::int64_t ScaledLogBaseTenOfTwo{301029995664};

// Models IEEE binary floating-point numbers (IEEE 754-2008,
// ISO/IEC/IEEE 60559.2011).  The first argument to this
// class template must be (or look like) an instance of Integer<>;
// the second specifies the number of effective bits (binary precision)
// in the fraction.
template <typename WORD, int PREC>
class Real : public common::RealDetails<PREC> {
public:
  using Word = WORD;
  static constexpr int binaryPrecision{PREC};
  using Details = common::RealDetails<PREC>;
  using Details::exponentBias;
  using Details::exponentBits;
  using Details::isImplicitMSB;
  using Details::maxExponent;
  using Details::significandBits;

  static constexpr int bits{Word::bits};
  static_assert(bits >= Details::bits);
  using Fraction = Integer<binaryPrecision>; // all bits made explicit

  template <typename W, int P> friend class Real;

  constexpr Real() {} // +0.0
  constexpr Real(const Real &) = default;
  constexpr Real(const Word &bits) : word_{bits} {}
  constexpr Real &operator=(const Real &) = default;
  constexpr Real &operator=(Real &&) = default;

  constexpr bool operator==(const Real &that) const {
    return word_ == that.word_;
  }

  // TODO: DIM, MAX, MIN, DPROD, FRACTION,
  // INT/NINT, NEAREST, OUT_OF_RANGE,
  // RRSPACING/SPACING, SCALE, SET_EXPONENT

  constexpr bool IsSignBitSet() const { return word_.BTEST(bits - 1); }
  constexpr bool IsNegative() const {
    return !IsNotANumber() && IsSignBitSet();
  }
  constexpr bool IsNotANumber() const {
    return Exponent() == maxExponent && !GetSignificand().IsZero();
  }
  constexpr bool IsQuietNaN() const {
    return Exponent() == maxExponent &&
        GetSignificand().BTEST(significandBits - 1);
  }
  constexpr bool IsSignalingNaN() const {
    return IsNotANumber() && !GetSignificand().BTEST(significandBits - 1);
  }
  constexpr bool IsInfinite() const {
    return Exponent() == maxExponent && GetSignificand().IsZero();
  }
  constexpr bool IsZero() const {
    return Exponent() == 0 && GetSignificand().IsZero();
  }
  constexpr bool IsSubnormal() const {
    return Exponent() == 0 && !GetSignificand().IsZero();
  }

  constexpr Real ABS() const { // non-arithmetic, no flags returned
    return {word_.IBCLR(bits - 1)};
  }
  constexpr Real SetSign(bool toNegative) const { // non-arithmetic
    if (toNegative) {
      return {word_.IBSET(bits - 1)};
    } else {
      return ABS();
    }
  }
  constexpr Real SIGN(const Real &x) const { return SetSign(x.IsSignBitSet()); }

  constexpr Real Negate() const { return {word_.IEOR(word_.MASKL(1))}; }

  Relation Compare(const Real &) const;
  ValueWithRealFlags<Real> Add(
      const Real &, Rounding rounding = defaultRounding) const;
  ValueWithRealFlags<Real> Subtract(
      const Real &y, Rounding rounding = defaultRounding) const {
    return Add(y.Negate(), rounding);
  }
  ValueWithRealFlags<Real> Multiply(
      const Real &, Rounding rounding = defaultRounding) const;
  ValueWithRealFlags<Real> Divide(
      const Real &, Rounding rounding = defaultRounding) const;

  // SQRT(x**2 + y**2) but computed so as to avoid spurious overflow
  // TODO: needed for CABS
  ValueWithRealFlags<Real> HYPOT(
      const Real &, Rounding rounding = defaultRounding) const;

  template <typename INT> constexpr INT EXPONENT() const {
    if (Exponent() == maxExponent) {
      return INT::HUGE();
    } else {
      return {UnbiasedExponent()};
    }
  }

  static constexpr Real EPSILON() {
    Real epsilon;
    epsilon.Normalize(
        false, exponentBias - binaryPrecision, Fraction::MASKL(1));
    return epsilon;
  }
  static constexpr Real HUGE() {
    Real huge;
    huge.Normalize(false, maxExponent - 1, Fraction::MASKR(binaryPrecision));
    return huge;
  }
  static constexpr Real TINY() {
    Real tiny;
    tiny.Normalize(false, 1, Fraction::MASKL(1)); // minimum *normal* number
    return tiny;
  }

  static constexpr int DIGITS{binaryPrecision};
  static constexpr int PRECISION{Details::decimalPrecision};
  static constexpr int RANGE{Details::decimalRange};
  static constexpr int MAXEXPONENT{maxExponent - 1 - exponentBias};
  static constexpr int MINEXPONENT{1 - exponentBias};

  constexpr Real FlushSubnormalToZero() const {
    if (IsSubnormal()) {
      return Real{};
    }
    return *this;
  }

  // TODO: Configurable NotANumber representations
  static constexpr Real NotANumber() {
    return {Word{maxExponent}
                .SHIFTL(significandBits)
                .IBSET(significandBits - 1)
                .IBSET(significandBits - 2)};
  }

  static constexpr Real Infinity(bool negative) {
    Word infinity{maxExponent};
    infinity = infinity.SHIFTL(significandBits);
    if (negative) {
      infinity = infinity.IBSET(infinity.bits - 1);
    }
    return {infinity};
  }

  template <typename INT>
  static ValueWithRealFlags<Real> FromInteger(
      const INT &n, Rounding rounding = defaultRounding) {
    bool isNegative{n.IsNegative()};
    INT absN{n};
    if (isNegative) {
      absN = n.Negate().value; // overflow is safe to ignore
    }
    int leadz{absN.LEADZ()};
    if (leadz >= absN.bits) {
      return {}; // all bits zero -> +0.0
    }
    ValueWithRealFlags<Real> result;
    int exponent{exponentBias + absN.bits - leadz - 1};
    int bitsNeeded{absN.bits - (leadz + isImplicitMSB)};
    int bitsLost{bitsNeeded - significandBits};
    if (bitsLost <= 0) {
      Fraction fraction{Fraction::ConvertUnsigned(absN).value};
      result.flags |= result.value.Normalize(
          isNegative, exponent, fraction.SHIFTL(-bitsLost));
    } else {
      Fraction fraction{Fraction::ConvertUnsigned(absN.SHIFTR(bitsLost)).value};
      result.flags |= result.value.Normalize(isNegative, exponent, fraction);
      RoundingBits roundingBits{absN, bitsLost};
      result.flags |= result.value.Round(rounding, roundingBits);
    }
    return result;
  }

  // Conversion to integer in the same real format (AINT(), ANINT())
  ValueWithRealFlags<Real> ToWholeNumber(
      common::RoundingMode = common::RoundingMode::ToZero) const;

  // Conversion to an integer (INT(), NINT(), FLOOR(), CEILING())
  template <typename INT>
  constexpr ValueWithRealFlags<INT> ToInteger(
      common::RoundingMode mode = common::RoundingMode::ToZero) const {
    ValueWithRealFlags<INT> result;
    if (IsNotANumber()) {
      result.flags.set(RealFlag::InvalidArgument);
      result.value = result.value.HUGE();
      return result;
    }
    ValueWithRealFlags<Real> intPart{ToWholeNumber(mode)};
    int exponent{intPart.value.Exponent()};
    result.flags.set(
        RealFlag::Overflow, exponent >= exponentBias + result.value.bits);
    result.flags |= intPart.flags;
    int shift{
        exponent - exponentBias - binaryPrecision + 1}; // positive -> left
    result.value =
        result.value.ConvertUnsigned(intPart.value.GetFraction().SHIFTR(-shift))
            .value.SHIFTL(shift);
    if (IsSignBitSet()) {
      auto negated{result.value.Negate()};
      result.value = negated.value;
      if (negated.overflow) {
        result.flags.set(RealFlag::Overflow);
      }
    }
    if (result.flags.test(RealFlag::Overflow)) {
      result.value =
          IsSignBitSet() ? result.value.MASKL(1) : result.value.HUGE();
    }
    return result;
  }

  template <typename A>
  static ValueWithRealFlags<Real> Convert(
      const A &x, Rounding rounding = defaultRounding) {
    ValueWithRealFlags<Real> result;
    if (x.IsNotANumber()) {
      result.flags.set(RealFlag::InvalidArgument);
      result.value = NotANumber();
      return result;
    }
    bool isNegative{x.IsNegative()};
    A absX{x};
    if (isNegative) {
      absX = x.Negate();
    }
    int exponent{exponentBias + x.UnbiasedExponent()};
    int bitsLost{A::binaryPrecision - binaryPrecision};
    if (exponent < 1) {
      bitsLost += 1 - exponent;
      exponent = 1;
    }
    typename A::Fraction xFraction{x.GetFraction()};
    if (bitsLost <= 0) {
      Fraction fraction{
          Fraction::ConvertUnsigned(xFraction).value.SHIFTL(-bitsLost)};
      result.flags |= result.value.Normalize(isNegative, exponent, fraction);
    } else {
      Fraction fraction{
          Fraction::ConvertUnsigned(xFraction.SHIFTR(bitsLost)).value};
      result.flags |= result.value.Normalize(isNegative, exponent, fraction);
      RoundingBits roundingBits{xFraction, bitsLost};
      result.flags |= result.value.Round(rounding, roundingBits);
    }
    return result;
  }

  constexpr Word RawBits() const { return word_; }

  // Extracts "raw" biased exponent field.
  constexpr int Exponent() const {
    return word_.IBITS(significandBits, exponentBits).ToUInt64();
  }

  // Extracts the fraction; any implied bit is made explicit.
  constexpr Fraction GetFraction() const {
    Fraction result{Fraction::ConvertUnsigned(word_).value};
    if constexpr (!isImplicitMSB) {
      return result;
    } else {
      int exponent{Exponent()};
      if (exponent > 0 && exponent < maxExponent) {
        return result.IBSET(significandBits);
      } else {
        return result.IBCLR(significandBits);
      }
    }
  }

  // Extracts unbiased exponent value.
  // Corrects the exponent value of a subnormal number.
  constexpr int UnbiasedExponent() const {
    int exponent{Exponent() - exponentBias};
    if (IsSubnormal()) {
      ++exponent;
    }
    return exponent;
  }

  static ValueWithRealFlags<Real> Read(
      const char *&, Rounding rounding = defaultRounding);
  std::string DumpHexadecimal() const;

  // Emits a character representation for an equivalent Fortran constant
  // or parenthesized constant expression that produces this value.
  llvm::raw_ostream &AsFortran(
      llvm::raw_ostream &, int kind, bool minimal = false) const;

private:
  using Significand = Integer<significandBits>; // no implicit bit

  constexpr Significand GetSignificand() const {
    return Significand::ConvertUnsigned(word_).value;
  }

  constexpr int CombineExponents(const Real &y, bool forDivide) const {
    int exponent = Exponent(), yExponent = y.Exponent();
    // A zero exponent field value has the same weight as 1.
    exponent += !exponent;
    yExponent += !yExponent;
    if (forDivide) {
      exponent += exponentBias - yExponent;
    } else {
      exponent += yExponent - exponentBias + 1;
    }
    return exponent;
  }

  static constexpr bool NextQuotientBit(
      Fraction &top, bool &msb, const Fraction &divisor) {
    bool greaterOrEqual{msb || top.CompareUnsigned(divisor) != Ordering::Less};
    if (greaterOrEqual) {
      top = top.SubtractSigned(divisor).value;
    }
    auto doubled{top.AddUnsigned(top)};
    top = doubled.value;
    msb = doubled.carry;
    return greaterOrEqual;
  }

  // Normalizes and marshals the fields of a floating-point number in place.
  // The value is a number, and a zero fraction means a zero value (i.e.,
  // a maximal exponent and zero fraction doesn't signify infinity, although
  // this member function will detect overflow and encode infinities).
  RealFlags Normalize(bool negative, int exponent, const Fraction &fraction,
      Rounding rounding = defaultRounding,
      RoundingBits *roundingBits = nullptr);

  // Rounds a result, if necessary, in place.
  RealFlags Round(Rounding, const RoundingBits &, bool multiply = false);

  static void NormalizeAndRound(ValueWithRealFlags<Real> &result,
      bool isNegative, int exponent, const Fraction &, Rounding, RoundingBits,
      bool multiply = false);

  Word word_{}; // an Integer<>
};

extern template class Real<Integer<16>, 11>; // IEEE half format
extern template class Real<Integer<16>, 8>; // the "other" half format
extern template class Real<Integer<32>, 24>; // IEEE single
extern template class Real<Integer<64>, 53>; // IEEE double
extern template class Real<Integer<80>, 64>; // 80387 extended precision
extern template class Real<Integer<128>, 113>; // IEEE quad
// N.B. No "double-double" support.
} // namespace Fortran::evaluate::value
#endif // FORTRAN_EVALUATE_REAL_H_
