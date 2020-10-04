//===-- lib/Decimal/binary-to-decimal.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "big-radix-floating-point.h"
#include "flang/Decimal/decimal.h"
#include <cassert>
#include <string>

namespace Fortran::decimal {

template <int PREC, int LOG10RADIX>
BigRadixFloatingPointNumber<PREC, LOG10RADIX>::BigRadixFloatingPointNumber(
    BinaryFloatingPointNumber<PREC> x, enum FortranRounding rounding)
    : rounding_{rounding} {
  bool negative{x.IsNegative()};
  if (x.IsZero()) {
    isNegative_ = negative;
    return;
  }
  if (negative) {
    x.Negate();
  }
  int twoPow{x.UnbiasedExponent()};
  twoPow -= x.bits - 1;
  if (!x.isImplicitMSB) {
    ++twoPow;
  }
  int lshift{x.exponentBits};
  if (twoPow <= -lshift) {
    twoPow += lshift;
    lshift = 0;
  } else if (twoPow < 0) {
    lshift += twoPow;
    twoPow = 0;
  }
  auto word{x.Fraction()};
  word <<= lshift;
  SetTo(word);
  isNegative_ = negative;

  // The significand is now encoded in *this as an integer (D) and
  // decimal exponent (E):  x = D * 10.**E * 2.**twoPow
  // twoPow can be positive or negative.
  // The goal now is to get twoPow up or down to zero, leaving us with
  // only decimal digits and decimal exponent.  This is done by
  // fast multiplications and divisions of D by 2 and 5.

  // (5*D) * 10.**E * 2.**twoPow -> D * 10.**(E+1) * 2.**(twoPow-1)
  for (; twoPow > 0 && IsDivisibleBy<5>(); --twoPow) {
    DivideBy<5>();
    ++exponent_;
  }

  int overflow{0};
  for (; twoPow >= 9; twoPow -= 9) {
    // D * 10.**E * 2.**twoPow -> (D*(2**9)) * 10.**E * 2.**(twoPow-9)
    overflow |= MultiplyBy<512>();
  }
  for (; twoPow >= 3; twoPow -= 3) {
    // D * 10.**E * 2.**twoPow -> (D*(2**3)) * 10.**E * 2.**(twoPow-3)
    overflow |= MultiplyBy<8>();
  }
  for (; twoPow > 0; --twoPow) {
    // D * 10.**E * 2.**twoPow -> (2*D) * 10.**E * 2.**(twoPow-1)
    overflow |= MultiplyBy<2>();
  }

  overflow |= DivideByPowerOfTwoInPlace(-twoPow);
  assert(overflow == 0);
  Normalize();
}

template <int PREC, int LOG10RADIX>
ConversionToDecimalResult
BigRadixFloatingPointNumber<PREC, LOG10RADIX>::ConvertToDecimal(char *buffer,
    std::size_t n, enum DecimalConversionFlags flags, int maxDigits) const {
  if (n < static_cast<std::size_t>(3 + digits_ * LOG10RADIX)) {
    return {nullptr, 0, 0, Overflow};
  }
  char *start{buffer};
  if (isNegative_) {
    *start++ = '-';
  } else if (flags & AlwaysSign) {
    *start++ = '+';
  }
  if (IsZero()) {
    *start++ = '0';
    *start = '\0';
    return {buffer, static_cast<std::size_t>(start - buffer), 0, Exact};
  }
  char *p{start};
  static_assert((LOG10RADIX % 2) == 0, "radix not a power of 100");
  static const char lut[] = "0001020304050607080910111213141516171819"
                            "2021222324252627282930313233343536373839"
                            "4041424344454647484950515253545556575859"
                            "6061626364656667686970717273747576777879"
                            "8081828384858687888990919293949596979899";
  // Treat the MSD specially: don't emit leading zeroes.
  Digit dig{digit_[digits_ - 1]};
  char stack[LOG10RADIX], *sp{stack};
  for (int k{0}; k < log10Radix; k += 2) {
    Digit newDig{dig / 100};
    auto d{static_cast<std::uint32_t>(dig) -
        std::uint32_t{100} * static_cast<std::uint32_t>(newDig)};
    dig = newDig;
    const char *q{lut + d + d};
    *sp++ = q[1];
    *sp++ = q[0];
  }
  while (sp > stack && sp[-1] == '0') {
    --sp;
  }
  while (sp > stack) {
    *p++ = *--sp;
  }
  for (int j{digits_ - 1}; j-- > 0;) {
    Digit dig{digit_[j]};
    char *reverse{p += log10Radix};
    for (int k{0}; k < log10Radix; k += 2) {
      Digit newDig{dig / 100};
      auto d{static_cast<std::uint32_t>(dig) -
          std::uint32_t{100} * static_cast<std::uint32_t>(newDig)};
      dig = newDig;
      const char *q{lut + d + d};
      *--reverse = q[1];
      *--reverse = q[0];
    }
  }
  // Adjust exponent so the effective decimal point is to
  // the left of the first digit.
  int expo = exponent_ + p - start;
  // Trim trailing zeroes.
  while (p[-1] == '0') {
    --p;
  }
  char *end{start + maxDigits};
  if (maxDigits == 0) {
    p = end;
  }
  if (p <= end) {
    *p = '\0';
    return {buffer, static_cast<std::size_t>(p - buffer), expo, Exact};
  } else {
    // Apply a digit limit, possibly with rounding.
    bool incr{false};
    switch (rounding_) {
    case RoundNearest:
      incr = *end > '5' ||
          (*end == '5' && (p > end + 1 || ((end[-1] - '0') & 1) != 0));
      break;
    case RoundUp:
      incr = !isNegative_;
      break;
    case RoundDown:
      incr = isNegative_;
      break;
    case RoundToZero:
      break;
    case RoundCompatible:
      incr = *end >= '5';
      break;
    }
    p = end;
    if (incr) {
      while (p > start && p[-1] == '9') {
        --p;
      }
      if (p == start) {
        *p++ = '1';
        ++expo;
      } else {
        ++p[-1];
      }
    }

    *p = '\0';
    return {buffer, static_cast<std::size_t>(p - buffer), expo, Inexact};
  }
}

template <int PREC, int LOG10RADIX>
bool BigRadixFloatingPointNumber<PREC, LOG10RADIX>::Mean(
    const BigRadixFloatingPointNumber &that) {
  while (digits_ < that.digits_) {
    digit_[digits_++] = 0;
  }
  int carry{0};
  for (int j{0}; j < that.digits_; ++j) {
    Digit v{digit_[j] + that.digit_[j] + carry};
    if (v >= radix) {
      digit_[j] = v - radix;
      carry = 1;
    } else {
      digit_[j] = v;
      carry = 0;
    }
  }
  if (carry != 0) {
    AddCarry(that.digits_, carry);
  }
  return DivideBy<2>() != 0;
}

template <int PREC, int LOG10RADIX>
void BigRadixFloatingPointNumber<PREC, LOG10RADIX>::Minimize(
    BigRadixFloatingPointNumber &&less, BigRadixFloatingPointNumber &&more) {
  int leastExponent{exponent_};
  if (less.exponent_ < leastExponent) {
    leastExponent = less.exponent_;
  }
  if (more.exponent_ < leastExponent) {
    leastExponent = more.exponent_;
  }
  while (exponent_ > leastExponent) {
    --exponent_;
    MultiplyBy<10>();
  }
  while (less.exponent_ > leastExponent) {
    --less.exponent_;
    less.MultiplyBy<10>();
  }
  while (more.exponent_ > leastExponent) {
    --more.exponent_;
    more.MultiplyBy<10>();
  }
  if (less.Mean(*this)) {
    less.AddCarry(); // round up
  }
  if (!more.Mean(*this)) {
    more.Decrement(); // round down
  }
  while (less.digits_ < more.digits_) {
    less.digit_[less.digits_++] = 0;
  }
  while (more.digits_ < less.digits_) {
    more.digit_[more.digits_++] = 0;
  }
  int digits{more.digits_};
  int same{0};
  while (same < digits &&
      less.digit_[digits - 1 - same] == more.digit_[digits - 1 - same]) {
    ++same;
  }
  if (same == digits) {
    return;
  }
  digits_ = same + 1;
  int offset{digits - digits_};
  exponent_ += offset * log10Radix;
  for (int j{0}; j < digits_; ++j) {
    digit_[j] = more.digit_[j + offset];
  }
  Digit least{less.digit_[offset]};
  Digit my{digit_[0]};
  while (true) {
    Digit q{my / 10u};
    Digit r{my - 10 * q};
    Digit lq{least / 10u};
    Digit lr{least - 10 * lq};
    if (r != 0 && lq == q) {
      Digit sub{(r - lr) >> 1};
      digit_[0] -= sub;
      break;
    } else {
      least = lq;
      my = q;
      DivideBy<10>();
      ++exponent_;
    }
  }
  Normalize();
}

template <int PREC>
ConversionToDecimalResult ConvertToDecimal(char *buffer, std::size_t size,
    enum DecimalConversionFlags flags, int digits,
    enum FortranRounding rounding, BinaryFloatingPointNumber<PREC> x) {
  if (x.IsNaN()) {
    return {"NaN", 3, 0, Invalid};
  } else if (x.IsInfinite()) {
    if (x.IsNegative()) {
      return {"-Inf", 4, 0, Exact};
    } else if (flags & AlwaysSign) {
      return {"+Inf", 4, 0, Exact};
    } else {
      return {"Inf", 3, 0, Exact};
    }
  } else {
    using Big = BigRadixFloatingPointNumber<PREC>;
    Big number{x, rounding};
    if ((flags & Minimize) && !x.IsZero()) {
      // To emit the fewest decimal digits necessary to represent the value
      // in such a way that decimal-to-binary conversion to the same format
      // with a fixed assumption about rounding will return the same binary
      // value, we also perform binary-to-decimal conversion on the two
      // binary values immediately adjacent to this one, use them to identify
      // the bounds of the range of decimal values that will map back to the
      // original binary value, and find a (not necessary unique) shortest
      // decimal sequence in that range.
      using Binary = typename Big::Real;
      Binary less{x};
      less.Previous();
      Binary more{x};
      if (!x.IsMaximalFiniteMagnitude()) {
        more.Next();
      }
      number.Minimize(Big{less, rounding}, Big{more, rounding});
    } else {
    }
    return number.ConvertToDecimal(buffer, size, flags, digits);
  }
}

template ConversionToDecimalResult ConvertToDecimal<8>(char *, std::size_t,
    enum DecimalConversionFlags, int, enum FortranRounding,
    BinaryFloatingPointNumber<8>);
template ConversionToDecimalResult ConvertToDecimal<11>(char *, std::size_t,
    enum DecimalConversionFlags, int, enum FortranRounding,
    BinaryFloatingPointNumber<11>);
template ConversionToDecimalResult ConvertToDecimal<24>(char *, std::size_t,
    enum DecimalConversionFlags, int, enum FortranRounding,
    BinaryFloatingPointNumber<24>);
template ConversionToDecimalResult ConvertToDecimal<53>(char *, std::size_t,
    enum DecimalConversionFlags, int, enum FortranRounding,
    BinaryFloatingPointNumber<53>);
template ConversionToDecimalResult ConvertToDecimal<64>(char *, std::size_t,
    enum DecimalConversionFlags, int, enum FortranRounding,
    BinaryFloatingPointNumber<64>);
template ConversionToDecimalResult ConvertToDecimal<113>(char *, std::size_t,
    enum DecimalConversionFlags, int, enum FortranRounding,
    BinaryFloatingPointNumber<113>);

extern "C" {
ConversionToDecimalResult ConvertFloatToDecimal(char *buffer, std::size_t size,
    enum DecimalConversionFlags flags, int digits,
    enum FortranRounding rounding, float x) {
  return Fortran::decimal::ConvertToDecimal(buffer, size, flags, digits,
      rounding, Fortran::decimal::BinaryFloatingPointNumber<24>(x));
}

ConversionToDecimalResult ConvertDoubleToDecimal(char *buffer, std::size_t size,
    enum DecimalConversionFlags flags, int digits,
    enum FortranRounding rounding, double x) {
  return Fortran::decimal::ConvertToDecimal(buffer, size, flags, digits,
      rounding, Fortran::decimal::BinaryFloatingPointNumber<53>(x));
}

#if __x86_64__ && !defined(_MSC_VER)
ConversionToDecimalResult ConvertLongDoubleToDecimal(char *buffer,
    std::size_t size, enum DecimalConversionFlags flags, int digits,
    enum FortranRounding rounding, long double x) {
  return Fortran::decimal::ConvertToDecimal(buffer, size, flags, digits,
      rounding, Fortran::decimal::BinaryFloatingPointNumber<64>(x));
}
#endif
}

template <int PREC, int LOG10RADIX>
template <typename STREAM>
STREAM &BigRadixFloatingPointNumber<PREC, LOG10RADIX>::Dump(STREAM &o) const {
  if (isNegative_) {
    o << '-';
  }
  o << "10**(" << exponent_ << ") * ...\n";
  for (int j{digits_}; --j >= 0;) {
    std::string str{std::to_string(digit_[j])};
    o << std::string(20 - str.size(), ' ') << str << " [" << j << ']';
    if (j + 1 == digitLimit_) {
      o << " (limit)";
    }
    o << '\n';
  }
  return o;
}
} // namespace Fortran::decimal
