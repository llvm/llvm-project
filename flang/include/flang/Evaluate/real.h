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
#include "flang/Evaluate/target.h"
#include <cinttypes>
#include <limits>
#include <string>

// Some environments, viz. glibc 2.17 and *BSD, allow the macro HUGE
// to leak out of <math.h>.
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
template <typename WORD, int PREC> class Real {
public:
  using Word = WORD;
  static constexpr int binaryPrecision{PREC};
  static constexpr common::RealCharacteristics realChars{PREC};
  static constexpr int exponentBias{realChars.exponentBias};
  static constexpr int exponentBits{realChars.exponentBits};
  static constexpr int isImplicitMSB{realChars.isImplicitMSB};
  static constexpr int maxExponent{realChars.maxExponent};
  static constexpr int significandBits{realChars.significandBits};

  static constexpr int bits{Word::bits};
  static_assert(bits >= realChars.bits);
  using Fraction = Integer<binaryPrecision>; // all bits made explicit

  template <typename W, int P> friend class Real;

  constexpr Real() {} // +0.0
  constexpr Real(const Real &) = default;
  constexpr Real(Real &&) = default;
  constexpr Real(const Word &bits) : word_{bits} {}
  constexpr Real &operator=(const Real &) = default;
  constexpr Real &operator=(Real &&) = default;

  constexpr bool operator==(const Real &that) const {
    return word_ == that.word_;
  }

  constexpr bool IsSignBitSet() const { return word_.BTEST(bits - 1); }
  constexpr bool IsNegative() const {
    return !IsNotANumber() && IsSignBitSet();
  }
  constexpr bool IsNotANumber() const {
    auto expo{Exponent()};
    auto sig{GetSignificand()};
    if constexpr (bits == 80) { // x87
      // 7FFF8000000000000000 is Infinity, not NaN, on 80387 & later.
      if (expo == maxExponent) {
        return sig != Significand{}.IBSET(63);
      } else {
        return expo != 0 && !sig.BTEST(63);
      }
    } else {
      return expo == maxExponent && !sig.IsZero();
    }
  }
  constexpr bool IsQuietNaN() const {
    auto expo{Exponent()};
    auto sig{GetSignificand()};
    if constexpr (bits == 80) { // x87
      if (expo == maxExponent) {
        return sig.IBITS(62, 2) == 3;
      } else {
        return expo != 0 && !sig.BTEST(63);
      }
    } else {
      return expo == maxExponent && sig.BTEST(significandBits - 1);
    }
  }
  constexpr bool IsSignalingNaN() const {
    auto expo{Exponent()};
    auto sig{GetSignificand()};
    if constexpr (bits == 80) { // x87
      return expo == maxExponent && sig != Significand{}.IBSET(63) &&
          sig.IBITS(62, 2) != 3;
    } else {
      return expo == maxExponent && !sig.IsZero() &&
          !sig.BTEST(significandBits - 1);
    }
  }
  constexpr bool IsInfinite() const {
    if constexpr (bits == 80) { // x87
      // 7FFF8000000000000000 is Infinity, not NaN, on 80387 & later.
      return Exponent() == maxExponent &&
          GetSignificand() == Significand{}.IBSET(63);
    } else {
      return Exponent() == maxExponent && GetSignificand().IsZero();
    }
  }
  constexpr bool IsFinite() const {
    auto expo{Exponent()};
    if constexpr (bits == 80) { // x87
      return expo != maxExponent && (expo == 0 || GetSignificand().BTEST(63));
    } else {
      return expo != maxExponent;
    }
  }
  constexpr bool IsZero() const {
    return Exponent() == 0 && GetSignificand().IsZero();
  }
  constexpr bool IsSubnormal() const {
    return Exponent() == 0 && !GetSignificand().IsZero();
  }
  constexpr bool IsNormal() const {
    return !(IsInfinite() || IsNotANumber() || IsSubnormal());
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
  ValueWithRealFlags<Real> Add(const Real &,
      Rounding rounding = TargetCharacteristics::defaultRounding) const;
  ValueWithRealFlags<Real> Subtract(const Real &y,
      Rounding rounding = TargetCharacteristics::defaultRounding) const {
    return Add(y.Negate(), rounding);
  }
  ValueWithRealFlags<Real> Multiply(const Real &,
      Rounding rounding = TargetCharacteristics::defaultRounding) const;
  ValueWithRealFlags<Real> Divide(const Real &,
      Rounding rounding = TargetCharacteristics::defaultRounding) const;

  ValueWithRealFlags<Real> SQRT(
      Rounding rounding = TargetCharacteristics::defaultRounding) const;
  // NEAREST(), IEEE_NEXT_AFTER(), IEEE_NEXT_UP(), and IEEE_NEXT_DOWN()
  ValueWithRealFlags<Real> NEAREST(bool upward) const;
  // HYPOT(x,y)=SQRT(x**2 + y**2) computed so as to avoid spurious
  // intermediate overflows.
  ValueWithRealFlags<Real> HYPOT(const Real &,
      Rounding rounding = TargetCharacteristics::defaultRounding) const;
  // DIM(X,Y) = MAX(X-Y, 0)
  ValueWithRealFlags<Real> DIM(const Real &,
      Rounding rounding = TargetCharacteristics::defaultRounding) const;
  // MOD(x,y) = x - AINT(x/y)*y (in the standard)
  // MODULO(x,y) = x - FLOOR(x/y)*y (in the standard)
  ValueWithRealFlags<Real> MOD(const Real &,
      Rounding rounding = TargetCharacteristics::defaultRounding) const;
  ValueWithRealFlags<Real> MODULO(const Real &,
      Rounding rounding = TargetCharacteristics::defaultRounding) const;

  template <typename INT> constexpr INT EXPONENT() const {
    if (Exponent() == maxExponent) {
      return INT::HUGE();
    } else if (IsZero()) {
      return {0};
    } else {
      return {UnbiasedExponent() + 1};
    }
  }

  static constexpr Real EPSILON() {
    Real epsilon;
    epsilon.Normalize(
        false, exponentBias + 1 - binaryPrecision, Fraction::MASKL(1));
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
  static constexpr int PRECISION{realChars.decimalPrecision};
  static constexpr int RANGE{realChars.decimalRange};
  static constexpr int MAXEXPONENT{maxExponent - exponentBias};
  static constexpr int MINEXPONENT{2 - exponentBias};
  Real RRSPACING() const;
  Real SPACING() const;
  Real SET_EXPONENT(std::int64_t) const;
  Real FRACTION() const;

  // SCALE(); also known as IEEE_SCALB and (in IEEE-754 '08) ScaleB.
  template <typename INT>
  ValueWithRealFlags<Real> SCALE(const INT &by,
      Rounding rounding = TargetCharacteristics::defaultRounding) const {
    // Normalize a fraction with just its LSB set and then multiply.
    // (Set the LSB, not the MSB, in case the scale factor needs to
    //  be subnormal.)
    constexpr auto adjust{exponentBias + binaryPrecision - 1};
    constexpr auto maxCoeffExpo{maxExponent + binaryPrecision - 1};
    auto expo{adjust + by.ToInt64()};
    RealFlags flags;
    int rMask{1};
    if (IsZero()) {
      expo = exponentBias; // ignore by, don't overflow
    } else if (expo > maxCoeffExpo) {
      if (Exponent() < exponentBias) {
        // Must implement with two multiplications
        return SCALE(INT{exponentBias})
            .value.SCALE(by.SubtractSigned(INT{exponentBias}).value, rounding);
      } else { // overflow
        expo = maxCoeffExpo;
      }
    } else if (expo < 0) {
      if (Exponent() > exponentBias) {
        // Must implement with two multiplications
        return SCALE(INT{-exponentBias})
            .value.SCALE(by.AddSigned(INT{exponentBias}).value, rounding);
      } else { // underflow to zero
        expo = 0;
        rMask = 0;
        flags.set(RealFlag::Underflow);
      }
    }
    Real twoPow;
    flags |=
        twoPow.Normalize(false, static_cast<int>(expo), Fraction::MASKR(rMask));
    ValueWithRealFlags<Real> result{Multiply(twoPow, rounding)};
    result.flags |= flags;
    return result;
  }

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

  static constexpr Real PositiveZero() { return Real{}; }

  static constexpr Real NegativeZero() { return {Word{}.MASKL(1)}; }

  static constexpr Real Infinity(bool negative) {
    Word infinity{maxExponent};
    infinity = infinity.SHIFTL(significandBits);
    if (negative) {
      infinity = infinity.IBSET(infinity.bits - 1);
    }
    if constexpr (bits == 80) { // x87
      // 7FFF8000000000000000 is Infinity, not NaN, on 80387 & later.
      infinity = infinity.IBSET(63);
    }
    return {infinity};
  }

  template <typename INT>
  static ValueWithRealFlags<Real> FromInteger(const INT &n,
      bool isUnsigned = false,
      Rounding rounding = TargetCharacteristics::defaultRounding) {
    bool isNegative{!isUnsigned && n.IsNegative()};
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
    result.flags |= intPart.flags;
    int exponent{intPart.value.Exponent()};
    // shift positive -> left shift, negative -> right shift
    int shift{exponent - exponentBias - binaryPrecision + 1};
    // Apply any right shift before moving to the result type
    auto rshifted{intPart.value.GetFraction().SHIFTR(-shift)};
    auto converted{result.value.ConvertUnsigned(rshifted)};
    if (converted.overflow) {
      result.flags.set(RealFlag::Overflow);
    }
    result.value = converted.value.SHIFTL(shift);
    if (converted.value.CompareUnsigned(result.value.SHIFTR(shift)) !=
        Ordering::Equal) {
      result.flags.set(RealFlag::Overflow);
    }
    if (IsSignBitSet()) {
      result.value = result.value.Negate().value;
    }
    if (!result.value.IsZero()) {
      if (IsSignBitSet() != result.value.IsNegative()) {
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
      const A &x, Rounding rounding = TargetCharacteristics::defaultRounding) {
    ValueWithRealFlags<Real> result;
    if (x.IsNotANumber()) {
      result.flags.set(RealFlag::InvalidArgument);
      result.value = NotANumber();
      return result;
    }
    bool isNegative{x.IsNegative()};
    if (x.IsInfinite()) {
      result.value = Infinity(isNegative);
      return result;
    }
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
  // Note that the result is one less than the EXPONENT intrinsic;
  // UnbiasedExponent(1.0) is 0, not 1.
  constexpr int UnbiasedExponent() const {
    int exponent{Exponent() - exponentBias};
    if (IsSubnormal()) {
      ++exponent;
    }
    return exponent;
  }

  static ValueWithRealFlags<Real> Read(const char *&,
      Rounding rounding = TargetCharacteristics::defaultRounding);
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
      Rounding rounding = TargetCharacteristics::defaultRounding,
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
extern template class Real<X87IntegerContainer, 64>; // 80387 extended precision
extern template class Real<Integer<128>, 113>; // IEEE quad
// N.B. No "double-double" support.
} // namespace Fortran::evaluate::value
#endif // FORTRAN_EVALUATE_REAL_H_
