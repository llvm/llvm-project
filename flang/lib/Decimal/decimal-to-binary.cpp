//===-- lib/Decimal/decimal-to-binary.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "big-radix-floating-point.h"
#include "flang/Common/bit-population-count.h"
#include "flang/Common/leading-zero-bit-count.h"
#include "flang/Decimal/binary-floating-point.h"
#include "flang/Decimal/decimal.h"
#include <cinttypes>
#include <cstring>
#include <ctype.h>

namespace Fortran::decimal {

template <int PREC, int LOG10RADIX>
bool BigRadixFloatingPointNumber<PREC, LOG10RADIX>::ParseNumber(
    const char *&p, bool &inexact, const char *end) {
  SetToZero();
  if (end && p >= end) {
    return false;
  }
  // Skip leading spaces
  for (; p != end && *p == ' '; ++p) {
  }
  if (p == end) {
    return false;
  }
  const char *q{p};
  isNegative_ = *q == '-';
  if (*q == '-' || *q == '+') {
    ++q;
  }
  const char *start{q};
  for (; q != end && *q == '0'; ++q) {
  }
  const char *firstDigit{q};
  for (; q != end && *q >= '0' && *q <= '9'; ++q) {
  }
  const char *point{nullptr};
  if (q != end && *q == '.') {
    point = q;
    for (++q; q != end && *q >= '0' && *q <= '9'; ++q) {
    }
  }
  if (q == start || (q == start + 1 && start == point)) {
    return false; // require at least one digit
  }
  // There's a valid number here; set the reference argument to point to
  // the first character afterward, which might be an exponent part.
  p = q;
  // Strip off trailing zeroes
  if (point) {
    while (q[-1] == '0') {
      --q;
    }
    if (q[-1] == '.') {
      point = nullptr;
      --q;
    }
  }
  if (!point) {
    while (q > firstDigit && q[-1] == '0') {
      --q;
      ++exponent_;
    }
  }
  // Trim any excess digits
  const char *limit{firstDigit + maxDigits * log10Radix + (point != nullptr)};
  if (q > limit) {
    inexact = true;
    if (point >= limit) {
      q = point;
      point = nullptr;
    }
    if (!point) {
      exponent_ += q - limit;
    }
    q = limit;
  }
  if (point) {
    exponent_ -= static_cast<int>(q - point - 1);
  }
  if (q == firstDigit) {
    exponent_ = 0; // all zeros
  }
  // Rack the decimal digits up into big Digits.
  for (auto times{radix}; q-- > firstDigit;) {
    if (*q != '.') {
      if (times == radix) {
        digit_[digits_++] = *q - '0';
        times = 10;
      } else {
        digit_[digits_ - 1] += times * (*q - '0');
        times *= 10;
      }
    }
  }
  // Look for an optional exponent field.
  if (p == end) {
    return true;
  }
  q = p;
  switch (*q) {
  case 'e':
  case 'E':
  case 'd':
  case 'D':
  case 'q':
  case 'Q': {
    if (++q == end) {
      break;
    }
    bool negExpo{*q == '-'};
    if (*q == '-' || *q == '+') {
      ++q;
    }
    if (q != end && *q >= '0' && *q <= '9') {
      int expo{0};
      for (; q != end && *q == '0'; ++q) {
      }
      const char *expDig{q};
      for (; q != end && *q >= '0' && *q <= '9'; ++q) {
        expo = 10 * expo + *q - '0';
      }
      if (q >= expDig + 8) {
        // There's a ridiculous number of nonzero exponent digits.
        // The decimal->binary conversion routine will cope with
        // returning 0 or Inf, but we must ensure that "expo" didn't
        // overflow back around to something legal.
        expo = 10 * Real::decimalRange;
        exponent_ = 0;
      }
      p = q; // exponent is valid; advance the termination pointer
      if (negExpo) {
        exponent_ -= expo;
      } else {
        exponent_ += expo;
      }
    }
  } break;
  default:
    break;
  }
  return true;
}

template <int PREC, int LOG10RADIX>
void BigRadixFloatingPointNumber<PREC,
    LOG10RADIX>::LoseLeastSignificantDigit() {
  Digit LSD{digit_[0]};
  for (int j{0}; j < digits_ - 1; ++j) {
    digit_[j] = digit_[j + 1];
  }
  digit_[digits_ - 1] = 0;
  bool incr{false};
  switch (rounding_) {
  case RoundNearest:
    incr = LSD > radix / 2 || (LSD == radix / 2 && digit_[0] % 2 != 0);
    break;
  case RoundUp:
    incr = LSD > 0 && !isNegative_;
    break;
  case RoundDown:
    incr = LSD > 0 && isNegative_;
    break;
  case RoundToZero:
    break;
  case RoundCompatible:
    incr = LSD >= radix / 2;
    break;
  }
  for (int j{0}; (digit_[j] += incr) == radix; ++j) {
    digit_[j] = 0;
  }
}

// This local utility class represents an unrounded nonnegative
// binary floating-point value with an unbiased (i.e., signed)
// binary exponent, an integer value (not a fraction) with an implied
// binary point to its *right*, and some guard bits for rounding.
template <int PREC> class IntermediateFloat {
public:
  static constexpr int precision{PREC};
  using IntType = common::HostUnsignedIntType<precision>;
  static constexpr IntType topBit{IntType{1} << (precision - 1)};
  static constexpr IntType mask{topBit + (topBit - 1)};

  IntermediateFloat() {}
  IntermediateFloat(const IntermediateFloat &) = default;

  // Assumes that exponent_ is valid on entry, and may increment it.
  // Returns the number of guard_ bits that have been determined.
  template <typename UINT> bool SetTo(UINT n) {
    static constexpr int nBits{CHAR_BIT * sizeof n};
    if constexpr (precision >= nBits) {
      value_ = n;
      guard_ = 0;
      return 0;
    } else {
      int shift{common::BitsNeededFor(n) - precision};
      if (shift <= 0) {
        value_ = n;
        guard_ = 0;
        return 0;
      } else {
        value_ = n >> shift;
        exponent_ += shift;
        n <<= nBits - shift;
        guard_ = (n >> (nBits - guardBits)) | ((n << guardBits) != 0);
        return shift;
      }
    }
  }

  void ShiftIn(int bit = 0) { value_ = value_ + value_ + bit; }
  bool IsFull() const { return value_ >= topBit; }
  void AdjustExponent(int by) { exponent_ += by; }
  void SetGuard(int g) {
    guard_ |= (static_cast<GuardType>(g & 6) << (guardBits - 3)) | (g & 1);
  }

  ConversionToBinaryResult<PREC> ToBinary(
      bool isNegative, FortranRounding) const;

private:
  static constexpr int guardBits{3}; // guard, round, sticky
  using GuardType = int;
  static constexpr GuardType oneHalf{GuardType{1} << (guardBits - 1)};

  IntType value_{0};
  GuardType guard_{0};
  int exponent_{0};
};

template <int PREC>
ConversionToBinaryResult<PREC> IntermediateFloat<PREC>::ToBinary(
    bool isNegative, FortranRounding rounding) const {
  using Binary = BinaryFloatingPointNumber<PREC>;
  // Create a fraction with a binary point to the left of the integer
  // value_, and bias the exponent.
  IntType fraction{value_};
  GuardType guard{guard_};
  int expo{exponent_ + Binary::exponentBias + (precision - 1)};
  while (expo < 1 && (fraction > 0 || guard > oneHalf)) {
    guard = (guard & 1) | (guard >> 1) |
        ((static_cast<GuardType>(fraction) & 1) << (guardBits - 1));
    fraction >>= 1;
    ++expo;
  }
  int flags{Exact};
  if (guard != 0) {
    flags |= Inexact;
  }
  if (fraction == 0 && guard <= oneHalf) {
    return {Binary{}, static_cast<enum ConversionResultFlags>(flags)};
  }
  // The value is nonzero; normalize it.
  while (fraction < topBit && expo > 1) {
    --expo;
    fraction = fraction * 2 + (guard >> (guardBits - 2));
    guard = (((guard >> (guardBits - 2)) & 1) << (guardBits - 1)) | (guard & 1);
  }
  // Apply rounding
  bool incr{false};
  switch (rounding) {
  case RoundNearest:
    incr = guard > oneHalf || (guard == oneHalf && (fraction & 1));
    break;
  case RoundUp:
    incr = guard != 0 && !isNegative;
    break;
  case RoundDown:
    incr = guard != 0 && isNegative;
    break;
  case RoundToZero:
    break;
  case RoundCompatible:
    incr = guard >= oneHalf;
    break;
  }
  if (incr) {
    if (fraction == mask) {
      // rounding causes a carry
      ++expo;
      fraction = topBit;
    } else {
      ++fraction;
    }
  }
  if (expo == 1 && fraction < topBit) {
    expo = 0; // subnormal
  }
  if (expo >= Binary::maxExponent) {
    expo = Binary::maxExponent; // Inf
    flags |= Overflow;
    fraction = 0;
  }
  using Raw = typename Binary::RawType;
  Raw raw = static_cast<Raw>(isNegative) << (Binary::bits - 1);
  raw |= static_cast<Raw>(expo) << Binary::significandBits;
  if constexpr (Binary::isImplicitMSB) {
    fraction &= ~topBit;
  }
  raw |= fraction;
  return {Binary(raw), static_cast<enum ConversionResultFlags>(flags)};
}

template <int PREC, int LOG10RADIX>
ConversionToBinaryResult<PREC>
BigRadixFloatingPointNumber<PREC, LOG10RADIX>::ConvertToBinary() {
  // On entry, *this holds a multi-precision integer value in a radix of a
  // large power of ten.  Its radix point is defined to be to the right of its
  // digits, and "exponent_" is the power of ten by which it is to be scaled.
  Normalize();
  if (digits_ == 0) { // zero value
    return {Real{SignBit()}};
  }
  // The value is not zero:  x = D. * 10.**E
  // Shift our perspective on the radix (& decimal) point so that
  // it sits to the *left* of the digits: i.e., x = .D * 10.**E
  exponent_ += digits_ * log10Radix;
  // Sanity checks for ridiculous exponents
  static constexpr int crazy{2 * Real::decimalRange + log10Radix};
  if (exponent_ < -crazy) { // underflow to +/-0.
    return {Real{SignBit()}, Inexact};
  } else if (exponent_ > crazy) { // overflow to +/-Inf.
    return {Real{Infinity()}, Overflow};
  }
  // Apply any negative decimal exponent by multiplication
  // by a power of two, adjusting the binary exponent to compensate.
  IntermediateFloat<PREC> f;
  while (exponent_ < log10Radix) {
    // x = 0.D * 10.**E * 2.**(f.ex) -> 512 * 0.D * 10.**E * 2.**(f.ex-9)
    f.AdjustExponent(-9);
    digitLimit_ = digits_;
    if (int carry{MultiplyWithoutNormalization<512>()}) {
      // x = c.D * 10.**E * 2.**(f.ex) -> .cD * 10.**(E+16) * 2.**(f.ex)
      PushCarry(carry);
      exponent_ += log10Radix;
    }
  }
  // Apply any positive decimal exponent greater than
  // is needed to treat the topmost digit as an integer
  // part by multiplying by 10 or 10000 repeatedly.
  while (exponent_ > log10Radix) {
    digitLimit_ = digits_;
    int carry;
    if (exponent_ >= log10Radix + 4) {
      // x = 0.D * 10.**E * 2.**(f.ex) -> 625 * .D * 10.**(E-4) * 2.**(f.ex+4)
      exponent_ -= 4;
      carry = MultiplyWithoutNormalization<(5 * 5 * 5 * 5)>();
      f.AdjustExponent(4);
    } else {
      // x = 0.D * 10.**E * 2.**(f.ex) -> 5 * .D * 10.**(E-1) * 2.**(f.ex+1)
      --exponent_;
      carry = MultiplyWithoutNormalization<5>();
      f.AdjustExponent(1);
    }
    if (carry != 0) {
      // x = c.D * 10.**E * 2.**(f.ex) -> .cD * 10.**(E+16) * 2.**(f.ex)
      PushCarry(carry);
      exponent_ += log10Radix;
    }
  }
  // So exponent_ is now log10Radix, meaning that the
  // MSD can be taken as an integer part and transferred
  // to the binary result.
  // x = .jD * 10.**16 * 2.**(f.ex) -> .D * j * 2.**(f.ex)
  int guardShift{f.SetTo(digit_[--digits_])};
  // Transfer additional bits until the result is normal.
  digitLimit_ = digits_;
  while (!f.IsFull()) {
    // x = ((b.D)/2) * j * 2.**(f.ex) -> .D * (2j + b) * 2.**(f.ex-1)
    f.AdjustExponent(-1);
    std::uint32_t carry = MultiplyWithoutNormalization<2>();
    f.ShiftIn(carry);
  }
  // Get the next few bits for rounding.  Allow for some guard bits
  // that may have already been set in f.SetTo() above.
  int guard{0};
  if (guardShift == 0) {
    guard = MultiplyWithoutNormalization<4>();
  } else if (guardShift == 1) {
    guard = MultiplyWithoutNormalization<2>();
  }
  guard = guard + guard + !IsZero();
  f.SetGuard(guard);
  return f.ToBinary(isNegative_, rounding_);
}

template <int PREC, int LOG10RADIX>
ConversionToBinaryResult<PREC>
BigRadixFloatingPointNumber<PREC, LOG10RADIX>::ConvertToBinary(
    const char *&p, const char *limit) {
  bool inexact{false};
  if (ParseNumber(p, inexact, limit)) {
    auto result{ConvertToBinary()};
    if (inexact) {
      result.flags =
          static_cast<enum ConversionResultFlags>(result.flags | Inexact);
    }
    return result;
  } else {
    // Could not parse a decimal floating-point number.  p has been
    // advanced over any leading spaces.
    if ((!limit || limit >= p + 3) && toupper(p[0]) == 'N' &&
        toupper(p[1]) == 'A' && toupper(p[2]) == 'N') {
      // NaN
      p += 3;
      if ((!limit || p < limit) && *p == '(') {
        int depth{1};
        do {
          ++p;
          if (limit && p >= limit) {
            // Invalid input
            return {Real{NaN()}, Invalid};
          } else if (*p == '(') {
            ++depth;
          } else if (*p == ')') {
            --depth;
          }
        } while (depth > 0);
        ++p;
      }
      return {Real{NaN()}};
    } else {
      // Try to parse Inf, maybe with a sign
      const char *q{p};
      if (!limit || q < limit) {
        isNegative_ = *q == '-';
        if (isNegative_ || *q == '+') {
          ++q;
        }
      }
      if ((!limit || limit >= q + 3) && toupper(q[0]) == 'I' &&
          toupper(q[1]) == 'N' && toupper(q[2]) == 'F') {
        if ((!limit || limit >= q + 8) && toupper(q[3]) == 'I' &&
            toupper(q[4]) == 'N' && toupper(q[5]) == 'I' &&
            toupper(q[6]) == 'T' && toupper(q[7]) == 'Y') {
          p = q + 8;
        } else {
          p = q + 3;
        }
        return {Real{Infinity()}};
      } else {
        // Invalid input
        return {Real{NaN()}, Invalid};
      }
    }
  }
}

template <int PREC>
ConversionToBinaryResult<PREC> ConvertToBinary(
    const char *&p, enum FortranRounding rounding, const char *end) {
  return BigRadixFloatingPointNumber<PREC>{rounding}.ConvertToBinary(p, end);
}

template ConversionToBinaryResult<8> ConvertToBinary<8>(
    const char *&, enum FortranRounding, const char *end);
template ConversionToBinaryResult<11> ConvertToBinary<11>(
    const char *&, enum FortranRounding, const char *end);
template ConversionToBinaryResult<24> ConvertToBinary<24>(
    const char *&, enum FortranRounding, const char *end);
template ConversionToBinaryResult<53> ConvertToBinary<53>(
    const char *&, enum FortranRounding, const char *end);
template ConversionToBinaryResult<64> ConvertToBinary<64>(
    const char *&, enum FortranRounding, const char *end);
template ConversionToBinaryResult<113> ConvertToBinary<113>(
    const char *&, enum FortranRounding, const char *end);

extern "C" {
enum ConversionResultFlags ConvertDecimalToFloat(
    const char **p, float *f, enum FortranRounding rounding) {
  auto result{Fortran::decimal::ConvertToBinary<24>(*p, rounding)};
  std::memcpy(reinterpret_cast<void *>(f),
      reinterpret_cast<const void *>(&result.binary), sizeof *f);
  return result.flags;
}
enum ConversionResultFlags ConvertDecimalToDouble(
    const char **p, double *d, enum FortranRounding rounding) {
  auto result{Fortran::decimal::ConvertToBinary<53>(*p, rounding)};
  std::memcpy(reinterpret_cast<void *>(d),
      reinterpret_cast<const void *>(&result.binary), sizeof *d);
  return result.flags;
}
enum ConversionResultFlags ConvertDecimalToLongDouble(
    const char **p, long double *ld, enum FortranRounding rounding) {
  auto result{Fortran::decimal::ConvertToBinary<64>(*p, rounding)};
  std::memcpy(reinterpret_cast<void *>(ld),
      reinterpret_cast<const void *>(&result.binary), sizeof *ld);
  return result.flags;
}
}
} // namespace Fortran::decimal
