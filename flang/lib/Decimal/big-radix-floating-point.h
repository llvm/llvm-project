//===-- lib/Decimal/big-radix-floating-point.h ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_DECIMAL_BIG_RADIX_FLOATING_POINT_H_
#define FORTRAN_DECIMAL_BIG_RADIX_FLOATING_POINT_H_

// This is a helper class for use in floating-point conversions between
// binary and decimal representations.  It holds a multiple-precision
// integer value using digits of a radix that is a large even power of ten
// (10,000,000,000,000,000 by default, 10**16).  These digits are accompanied
// by a signed exponent that denotes multiplication by a power of ten.
// The effective radix point is to the right of the digits (i.e., they do
// not represent a fraction).
//
// The operations supported by this class are limited to those required
// for conversions between binary and decimal representations; it is not
// a general-purpose facility.

#include "flang/Common/bit-population-count.h"
#include "flang/Common/leading-zero-bit-count.h"
#include "flang/Common/uint128.h"
#include "flang/Decimal/binary-floating-point.h"
#include "flang/Decimal/decimal.h"
#include <cinttypes>
#include <limits>
#include <type_traits>

namespace Fortran::decimal {

static constexpr std::uint64_t TenToThe(int power) {
  return power <= 0 ? 1 : 10 * TenToThe(power - 1);
}

// 10**(LOG10RADIX + 3) must be < 2**wordbits, and LOG10RADIX must be
// even, so that pairs of decimal digits do not straddle Digits.
// So LOG10RADIX must be 16 or 6.
template <int PREC, int LOG10RADIX = 16> class BigRadixFloatingPointNumber {
public:
  using Real = BinaryFloatingPointNumber<PREC>;
  static constexpr int log10Radix{LOG10RADIX};

private:
  static constexpr std::uint64_t uint64Radix{TenToThe(log10Radix)};
  static constexpr int minDigitBits{
      64 - common::LeadingZeroBitCount(uint64Radix)};
  using Digit = common::HostUnsignedIntType<minDigitBits>;
  static constexpr Digit radix{uint64Radix};
  static_assert(radix < std::numeric_limits<Digit>::max() / 1000,
      "radix is somehow too big");
  static_assert(radix > std::numeric_limits<Digit>::max() / 10000,
      "radix is somehow too small");

  // The base-2 logarithm of the least significant bit that can arise
  // in a subnormal IEEE floating-point number.
  static constexpr int minLog2AnyBit{
      -Real::exponentBias - Real::binaryPrecision};

  // The number of Digits needed to represent the smallest subnormal.
  static constexpr int maxDigits{3 - minLog2AnyBit / log10Radix};

public:
  explicit BigRadixFloatingPointNumber(
      enum FortranRounding rounding = RoundNearest)
      : rounding_{rounding} {}

  // Converts a binary floating point value.
  explicit BigRadixFloatingPointNumber(
      Real, enum FortranRounding = RoundNearest);

  BigRadixFloatingPointNumber &SetToZero() {
    isNegative_ = false;
    digits_ = 0;
    exponent_ = 0;
    return *this;
  }

  // Converts decimal floating-point to binary.
  ConversionToBinaryResult<PREC> ConvertToBinary();

  // Parses and converts to binary.  Handles leading spaces,
  // "NaN", & optionally-signed "Inf".  Does not skip internal
  // spaces.
  // The argument is a reference to a pointer that is left
  // pointing to the first character that wasn't parsed.
  ConversionToBinaryResult<PREC> ConvertToBinary(
      const char *&, const char *end = nullptr);

  // Formats a decimal floating-point number to a user buffer.
  // May emit "NaN" or "Inf", or an possibly-signed integer.
  // No decimal point is written, but if it were, it would be
  // after the last digit; the effective decimal exponent is
  // returned as part of the result structure so that it can be
  // formatted by the client.
  ConversionToDecimalResult ConvertToDecimal(
      char *, std::size_t, enum DecimalConversionFlags, int digits) const;

  // Discard decimal digits not needed to distinguish this value
  // from the decimal encodings of two others (viz., the nearest binary
  // floating-point numbers immediately below and above this one).
  // The last decimal digit may not be uniquely determined in all
  // cases, and will be the mean value when that is so (e.g., if
  // last decimal digit values 6-8 would all work, it'll be a 7).
  // This minimization necessarily assumes that the value will be
  // emitted and read back into the same (or less precise) format
  // with default rounding to the nearest value.
  void Minimize(
      BigRadixFloatingPointNumber &&less, BigRadixFloatingPointNumber &&more);

  template <typename STREAM> STREAM &Dump(STREAM &) const;

private:
  BigRadixFloatingPointNumber(const BigRadixFloatingPointNumber &that)
      : digits_{that.digits_}, exponent_{that.exponent_},
        isNegative_{that.isNegative_}, rounding_{that.rounding_} {
    for (int j{0}; j < digits_; ++j) {
      digit_[j] = that.digit_[j];
    }
  }

  bool IsZero() const {
    // Don't assume normalization.
    for (int j{0}; j < digits_; ++j) {
      if (digit_[j] != 0) {
        return false;
      }
    }
    return true;
  }

  // Predicate: true when 10*value would cause a carry.
  // (When this happens during decimal-to-binary conversion,
  // there are more digits in the input string than can be
  // represented precisely.)
  bool IsFull() const {
    return digits_ == digitLimit_ && digit_[digits_ - 1] >= radix / 10;
  }

  // Sets *this to an unsigned integer value.
  // Returns any remainder.
  template <typename UINT> UINT SetTo(UINT n) {
    static_assert(
        std::is_same_v<UINT, common::uint128_t> || std::is_unsigned_v<UINT>);
    SetToZero();
    while (n != 0) {
      auto q{n / 10u};
      if (n != q * 10) {
        break;
      }
      ++exponent_;
      n = q;
    }
    if constexpr (sizeof n < sizeof(Digit)) {
      if (n != 0) {
        digit_[digits_++] = n;
      }
      return 0;
    } else {
      while (n != 0 && digits_ < digitLimit_) {
        auto q{n / radix};
        digit_[digits_++] = static_cast<Digit>(n - q * radix);
        n = q;
      }
      return n;
    }
  }

  int RemoveLeastOrderZeroDigits() {
    int remove{0};
    if (digits_ > 0 && digit_[0] == 0) {
      while (remove < digits_ && digit_[remove] == 0) {
        ++remove;
      }
      if (remove >= digits_) {
        digits_ = 0;
      } else if (remove > 0) {
#if defined __GNUC__ && __GNUC__ < 8
        // (&& j + remove < maxDigits) was added to avoid GCC < 8 build failure
        // on -Werror=array-bounds. This can be removed if -Werror is disable.
        for (int j{0}; j + remove < digits_ && (j + remove < maxDigits); ++j) {
#else
        for (int j{0}; j + remove < digits_; ++j) {
#endif
          digit_[j] = digit_[j + remove];
        }
        digits_ -= remove;
      }
    }
    return remove;
  }

  void RemoveLeadingZeroDigits() {
    while (digits_ > 0 && digit_[digits_ - 1] == 0) {
      --digits_;
    }
  }

  void Normalize() {
    RemoveLeadingZeroDigits();
    exponent_ += RemoveLeastOrderZeroDigits() * log10Radix;
  }

  // This limited divisibility test only works for even divisors of the radix,
  // which is fine since it's only ever used with 2 and 5.
  template <int N> bool IsDivisibleBy() const {
    static_assert(N > 1 && radix % N == 0, "bad modulus");
    return digits_ == 0 || (digit_[0] % N) == 0;
  }

  template <unsigned DIVISOR> int DivideBy() {
    Digit remainder{0};
    for (int j{digits_ - 1}; j >= 0; --j) {
      Digit q{digit_[j] / DIVISOR};
      Digit nrem{digit_[j] - DIVISOR * q};
      digit_[j] = q + (radix / DIVISOR) * remainder;
      remainder = nrem;
    }
    return remainder;
  }

  void DivideByPowerOfTwo(int twoPow) { // twoPow <= log10Radix
    Digit remainder{0};
    auto mask{(Digit{1} << twoPow) - 1};
    auto coeff{radix >> twoPow};
    for (int j{digits_ - 1}; j >= 0; --j) {
      auto nrem{digit_[j] & mask};
      digit_[j] = (digit_[j] >> twoPow) + coeff * remainder;
      remainder = nrem;
    }
  }

  // Returns true on overflow
  bool DivideByPowerOfTwoInPlace(int twoPow) {
    if (digits_ > 0) {
      while (twoPow > 0) {
        int chunk{twoPow > log10Radix ? log10Radix : twoPow};
        if ((digit_[0] & ((Digit{1} << chunk) - 1)) == 0) {
          DivideByPowerOfTwo(chunk);
          twoPow -= chunk;
          continue;
        }
        twoPow -= chunk;
        if (digit_[digits_ - 1] >> chunk != 0) {
          if (digits_ == digitLimit_) {
            return true; // overflow
          }
          digit_[digits_++] = 0;
        }
        auto remainder{digit_[digits_ - 1]};
        exponent_ -= log10Radix;
        auto coeff{radix >> chunk}; // precise; radix is (5*2)**log10Radix
        auto mask{(Digit{1} << chunk) - 1};
        for (int j{digits_ - 1}; j >= 1; --j) {
          digit_[j] = (digit_[j - 1] >> chunk) + coeff * remainder;
          remainder = digit_[j - 1] & mask;
        }
        digit_[0] = coeff * remainder;
      }
    }
    return false; // no overflow
  }

  int AddCarry(int position = 0, int carry = 1) {
    for (; position < digits_; ++position) {
      Digit v{digit_[position] + carry};
      if (v < radix) {
        digit_[position] = v;
        return 0;
      }
      digit_[position] = v - radix;
      carry = 1;
    }
    if (digits_ < digitLimit_) {
      digit_[digits_++] = carry;
      return 0;
    }
    Normalize();
    if (digits_ < digitLimit_) {
      digit_[digits_++] = carry;
      return 0;
    }
    return carry;
  }

  void Decrement() {
    for (int j{0}; digit_[j]-- == 0; ++j) {
      digit_[j] = radix - 1;
    }
  }

  template <int N> int MultiplyByHelper(int carry = 0) {
    for (int j{0}; j < digits_; ++j) {
      auto v{N * digit_[j] + carry};
      carry = v / radix;
      digit_[j] = v - carry * radix; // i.e., v % radix
    }
    return carry;
  }

  template <int N> int MultiplyBy(int carry = 0) {
    if (int newCarry{MultiplyByHelper<N>(carry)}) {
      return AddCarry(digits_, newCarry);
    } else {
      return 0;
    }
  }

  template <int N> int MultiplyWithoutNormalization() {
    if (int carry{MultiplyByHelper<N>(0)}) {
      if (digits_ < digitLimit_) {
        digit_[digits_++] = carry;
        return 0;
      } else {
        return carry;
      }
    } else {
      return 0;
    }
  }

  void LoseLeastSignificantDigit(); // with rounding

  void PushCarry(int carry) {
    if (digits_ == maxDigits && RemoveLeastOrderZeroDigits() == 0) {
      LoseLeastSignificantDigit();
      digit_[digits_ - 1] += carry;
    } else {
      digit_[digits_++] = carry;
    }
  }

  // Adds another number and then divides by two.
  // Assumes same exponent and sign.
  // Returns true when the the result has effectively been rounded down.
  bool Mean(const BigRadixFloatingPointNumber &);

  // Parses a floating-point number; leaves the pointer reference
  // argument pointing at the next character after what was recognized.
  // The "end" argument can be left null if the caller is sure that the
  // string is properly terminated with an addressable character that
  // can't be in a valid floating-point character.
  bool ParseNumber(const char *&, bool &inexact, const char *end);

  using Raw = typename Real::RawType;
  constexpr Raw SignBit() const { return Raw{isNegative_} << (Real::bits - 1); }
  constexpr Raw Infinity() const {
    Raw result{static_cast<Raw>(Real::maxExponent)};
    result <<= Real::significandBits;
    result |= SignBit();
    if constexpr (Real::bits == 80) { // x87
      result |= Raw{1} << 63;
    }
    return result;
  }
  constexpr Raw NaN(bool isQuiet = true) {
    Raw result{Real::maxExponent};
    result <<= Real::significandBits;
    result |= SignBit();
    if constexpr (Real::bits == 80) { // x87
      result |= Raw{isQuiet ? 3u : 2u} << 62;
    } else {
      Raw quiet{isQuiet ? Raw{2} : Raw{1}};
      quiet <<= Real::significandBits - 2;
      result |= quiet;
    }
    return result;
  }

  Digit digit_[maxDigits]; // in little-endian order: digit_[0] is LSD
  int digits_{0}; // # of elements in digit_[] array; zero when zero
  int digitLimit_{maxDigits}; // precision clamp
  int exponent_{0}; // signed power of ten
  bool isNegative_{false};
  enum FortranRounding rounding_ { RoundNearest };
};
} // namespace Fortran::decimal
#endif
