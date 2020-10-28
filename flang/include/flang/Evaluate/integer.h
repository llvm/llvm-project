//===-- include/flang/Evaluate/integer.h ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_EVALUATE_INTEGER_H_
#define FORTRAN_EVALUATE_INTEGER_H_

// Emulates binary integers of an arbitrary (but fixed) bit size for use
// when the host C++ environment does not support that size or when the
// full suite of Fortran's integer intrinsic scalar functions are needed.
// The data model is typeless, so signed* and unsigned operations
// are distinguished from each other with distinct member function interfaces.
// (*"Signed" here means two's-complement, just to be clear.  Ones'-complement
// and signed-magnitude encodings appear to be extinct in 2018.)

#include "flang/Common/bit-population-count.h"
#include "flang/Common/leading-zero-bit-count.h"
#include "flang/Evaluate/common.h"
#include <cinttypes>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>

// Some environments, viz. clang on Darwin, allow the macro HUGE
// to leak out of <math.h> even when it is never directly included.
#undef HUGE

namespace Fortran::evaluate::value {

// Implements an integer as an assembly of smaller host integer parts
// that constitute the digits of a large-radix fixed-point number.
// For best performance, the type of these parts should be half of the
// size of the largest efficient integer supported by the host processor.
// These parts are stored in either little- or big-endian order, which can
// match that of the host's endianness or not; but if the ordering matches
// that of the host, raw host data can be overlaid with a properly configured
// instance of this class and used in situ.
// To facilitate exhaustive testing of what would otherwise be more rare
// edge cases, this class template may be configured to use other part
// types &/or partial fields in the parts.  The radix (i.e., the number
// of possible values in a part), however, must be a power of two; this
// template class is not generalized to enable, say, decimal arithmetic.
// Member functions that correspond to Fortran intrinsic functions are
// named accordingly in ALL CAPS so that they can be referenced easily in
// the language standard.
template <int BITS, bool IS_LITTLE_ENDIAN = isHostLittleEndian,
    int PARTBITS = BITS <= 32 ? BITS : 32,
    typename PART = HostUnsignedInt<PARTBITS>,
    typename BIGPART = HostUnsignedInt<PARTBITS * 2>>
class Integer {
public:
  static constexpr int bits{BITS};
  static constexpr int partBits{PARTBITS};
  using Part = PART;
  using BigPart = BIGPART;
  static_assert(std::is_integral_v<Part>);
  static_assert(std::is_unsigned_v<Part>);
  static_assert(std::is_integral_v<BigPart>);
  static_assert(std::is_unsigned_v<BigPart>);
  static_assert(CHAR_BIT * sizeof(BigPart) >= 2 * partBits);
  static constexpr bool littleEndian{IS_LITTLE_ENDIAN};

private:
  static constexpr int maxPartBits{CHAR_BIT * sizeof(Part)};
  static_assert(partBits > 0 && partBits <= maxPartBits);
  static constexpr int extraPartBits{maxPartBits - partBits};
  static constexpr int parts{(bits + partBits - 1) / partBits};
  static_assert(parts >= 1);
  static constexpr int extraTopPartBits{
      extraPartBits + (parts * partBits) - bits};
  static constexpr int topPartBits{maxPartBits - extraTopPartBits};
  static_assert(topPartBits > 0 && topPartBits <= partBits);
  static_assert((parts - 1) * partBits + topPartBits == bits);
  static constexpr Part partMask{static_cast<Part>(~0) >> extraPartBits};
  static constexpr Part topPartMask{static_cast<Part>(~0) >> extraTopPartBits};

public:
  // Some types used for member function results
  struct ValueWithOverflow {
    Integer value;
    bool overflow;
  };

  struct ValueWithCarry {
    Integer value;
    bool carry;
  };

  struct Product {
    bool SignedMultiplicationOverflowed() const {
      return lower.IsNegative() ? (upper.POPCNT() != bits) : !upper.IsZero();
    }
    Integer upper, lower;
  };

  struct QuotientWithRemainder {
    Integer quotient, remainder;
    bool divisionByZero, overflow;
  };

  struct PowerWithErrors {
    Integer power;
    bool divisionByZero{false}, overflow{false}, zeroToZero{false};
  };

  // Constructors and value-generating static functions
  constexpr Integer() { Clear(); } // default constructor: zero
  constexpr Integer(const Integer &) = default;
  constexpr Integer(Integer &&) = default;

  // C++'s integral types can all be converted to Integer
  // with silent truncation.
  template <typename INT, typename = std::enable_if_t<std::is_integral_v<INT>>>
  constexpr Integer(INT n) {
    constexpr int nBits = CHAR_BIT * sizeof n;
    if constexpr (nBits < partBits) {
      if constexpr (std::is_unsigned_v<INT>) {
        // Zero-extend an unsigned smaller value.
        SetLEPart(0, n);
        for (int j{1}; j < parts; ++j) {
          SetLEPart(j, 0);
        }
      } else {
        // n has a signed type smaller than the usable
        // bits in a Part.
        // Avoid conversions that change both size and sign.
        using SignedPart = std::make_signed_t<Part>;
        Part p = static_cast<SignedPart>(n);
        SetLEPart(0, p);
        if constexpr (parts > 1) {
          Part signExtension = static_cast<SignedPart>(-(n < 0));
          for (int j{1}; j < parts; ++j) {
            SetLEPart(j, signExtension);
          }
        }
      }
    } else {
      // n has some integral type no smaller than the usable
      // bits in a Part.
      // Ensure that all shifts are smaller than a whole word.
      if constexpr (std::is_unsigned_v<INT>) {
        for (int j{0}; j < parts; ++j) {
          SetLEPart(j, static_cast<Part>(n));
          if constexpr (nBits > partBits) {
            n >>= partBits;
          } else {
            n = 0;
          }
        }
      } else {
        INT signExtension{-(n < 0)};
        static_assert(nBits >= partBits);
        if constexpr (nBits > partBits) {
          signExtension <<= nBits - partBits;
          for (int j{0}; j < parts; ++j) {
            SetLEPart(j, static_cast<Part>(n));
            n >>= partBits;
            n |= signExtension;
          }
        } else {
          SetLEPart(0, static_cast<Part>(n));
          for (int j{1}; j < parts; ++j) {
            SetLEPart(j, static_cast<Part>(signExtension));
          }
        }
      }
    }
  }

  constexpr Integer &operator=(const Integer &) = default;

  constexpr bool operator<(const Integer &that) const {
    return CompareSigned(that) == Ordering::Less;
  }
  constexpr bool operator<=(const Integer &that) const {
    return CompareSigned(that) != Ordering::Greater;
  }
  constexpr bool operator==(const Integer &that) const {
    return CompareSigned(that) == Ordering::Equal;
  }
  constexpr bool operator!=(const Integer &that) const {
    return !(*this == that);
  }
  constexpr bool operator>=(const Integer &that) const {
    return CompareSigned(that) != Ordering::Less;
  }
  constexpr bool operator>(const Integer &that) const {
    return CompareSigned(that) == Ordering::Greater;
  }

  // Left-justified mask (e.g., MASKL(1) has only its sign bit set)
  static constexpr Integer MASKL(int places) {
    if (places <= 0) {
      return {};
    } else if (places >= bits) {
      return MASKR(bits);
    } else {
      return MASKR(bits - places).NOT();
    }
  }

  // Right-justified mask (e.g., MASKR(1) == 1, MASKR(2) == 3, &c.)
  static constexpr Integer MASKR(int places) {
    Integer result{nullptr};
    int j{0};
    for (; j + 1 < parts && places >= partBits; ++j, places -= partBits) {
      result.LEPart(j) = partMask;
    }
    if (places > 0) {
      if (j + 1 < parts) {
        result.LEPart(j++) = partMask >> (partBits - places);
      } else if (j + 1 == parts) {
        if (places >= topPartBits) {
          result.LEPart(j++) = topPartMask;
        } else {
          result.LEPart(j++) = topPartMask >> (topPartBits - places);
        }
      }
    }
    for (; j < parts; ++j) {
      result.LEPart(j) = 0;
    }
    return result;
  }

  static constexpr ValueWithOverflow Read(
      const char *&pp, std::uint64_t base = 10, bool isSigned = false) {
    Integer result;
    bool overflow{false};
    const char *p{pp};
    while (*p == ' ' || *p == '\t') {
      ++p;
    }
    bool negate{*p == '-'};
    if (negate || *p == '+') {
      while (*++p == ' ' || *p == '\t') {
      }
    }
    Integer radix{base};
    // This code makes assumptions about local contiguity in regions of the
    // character set and only works up to base 36.  These assumptions hold
    // for all current combinations of surviving character sets (ASCII, UTF-8,
    // EBCDIC) and the bases used in Fortran source and formatted I/O
    // (viz., 2, 8, 10, & 16).  But: management thought that a disclaimer
    // might be needed here to warn future users of this code about these
    // assumptions, so here you go, future programmer in some postapocalyptic
    // hellscape, and best of luck with the inexorable killer robots.
    for (; std::uint64_t digit = *p; ++p) {
      if (digit >= '0' && digit <= '9' && digit < '0' + base) {
        digit -= '0';
      } else if (base > 10 && digit >= 'A' && digit < 'A' + base - 10) {
        digit -= 'A' - 10;
      } else if (base > 10 && digit >= 'a' && digit < 'a' + base - 10) {
        digit -= 'a' - 10;
      } else {
        break;
      }
      Product shifted{result.MultiplyUnsigned(radix)};
      overflow |= !shifted.upper.IsZero();
      ValueWithCarry next{shifted.lower.AddUnsigned(Integer{digit})};
      overflow |= next.carry;
      result = next.value;
    }
    pp = p;
    if (negate) {
      result = result.Negate().value;
      overflow |= isSigned && !result.IsNegative() && !result.IsZero();
    } else {
      overflow |= isSigned && result.IsNegative();
    }
    return {result, overflow};
  }

  template <typename FROM>
  static constexpr ValueWithOverflow ConvertUnsigned(const FROM &that) {
    std::uint64_t field{that.ToUInt64()};
    ValueWithOverflow result{field, false};
    if constexpr (bits < 64) {
      result.overflow = (field >> bits) != 0;
    }
    for (int j{64}; j < that.bits && !result.overflow; j += 64) {
      field = that.SHIFTR(j).ToUInt64();
      if (bits <= j) {
        result.overflow = field != 0;
      } else {
        result.value = result.value.IOR(Integer{field}.SHIFTL(j));
        if (bits < j + 64) {
          result.overflow = (field >> (bits - j)) != 0;
        }
      }
    }
    return result;
  }

  template <typename FROM>
  static constexpr ValueWithOverflow ConvertSigned(const FROM &that) {
    ValueWithOverflow result{ConvertUnsigned(that)};
    if constexpr (bits > FROM::bits) {
      if (that.IsNegative()) {
        result.value = result.value.IOR(MASKL(bits - FROM::bits));
      }
      result.overflow = false;
    } else if constexpr (bits < FROM::bits) {
      auto back{FROM::template ConvertSigned(result.value)};
      result.overflow = back.value.CompareUnsigned(that) != Ordering::Equal;
    }
    return result;
  }

  std::string UnsignedDecimal() const {
    if constexpr (bits < 4) {
      char digit = '0' + ToUInt64();
      return {digit};
    } else if (IsZero()) {
      return {'0'};
    } else {
      QuotientWithRemainder qr{DivideUnsigned(10)};
      char digit = '0' + qr.remainder.ToUInt64();
      if (qr.quotient.IsZero()) {
        return {digit};
      } else {
        return qr.quotient.UnsignedDecimal() + digit;
      }
    }
  }

  std::string SignedDecimal() const {
    if (IsNegative()) {
      return std::string{'-'} + Negate().value.UnsignedDecimal();
    } else {
      return UnsignedDecimal();
    }
  }

  // Omits a leading "0x".
  std::string Hexadecimal() const {
    std::string result;
    int digits{(bits + 3) >> 2};
    for (int j{0}; j < digits; ++j) {
      int pos{(digits - 1 - j) * 4};
      char nybble = IBITS(pos, 4).ToUInt64();
      if (nybble != 0 || !result.empty() || j + 1 == digits) {
        char digit = '0' + nybble;
        if (digit > '9') {
          digit += 'a' - ('9' + 1);
        }
        result += digit;
      }
    }
    return result;
  }

  static constexpr int DIGITS{bits - 1}; // don't count the sign bit
  static constexpr Integer HUGE() { return MASKR(bits - 1); }
  static constexpr int RANGE{// in the sense of SELECTED_INT_KIND
      // This magic value is LOG10(2.)*1E12.
      static_cast<int>(((bits - 1) * 301029995664) / 1000000000000)};

  constexpr bool IsZero() const {
    for (int j{0}; j < parts; ++j) {
      if (part_[j] != 0) {
        return false;
      }
    }
    return true;
  }

  constexpr bool IsNegative() const {
    return (LEPart(parts - 1) >> (topPartBits - 1)) & 1;
  }

  constexpr Ordering CompareToZeroSigned() const {
    if (IsNegative()) {
      return Ordering::Less;
    } else if (IsZero()) {
      return Ordering::Equal;
    } else {
      return Ordering::Greater;
    }
  }

  // Count the number of contiguous most-significant bit positions
  // that are clear.
  constexpr int LEADZ() const {
    if (LEPart(parts - 1) != 0) {
      int lzbc{common::LeadingZeroBitCount(LEPart(parts - 1))};
      return lzbc - extraTopPartBits;
    }
    int upperZeroes{topPartBits};
    for (int j{1}; j < parts; ++j) {
      if (Part p{LEPart(parts - 1 - j)}) {
        int lzbc{common::LeadingZeroBitCount(p)};
        return upperZeroes + lzbc - extraPartBits;
      }
      upperZeroes += partBits;
    }
    return bits;
  }

  // Count the number of bit positions that are set.
  constexpr int POPCNT() const {
    int count{0};
    for (int j{0}; j < parts; ++j) {
      count += common::BitPopulationCount(part_[j]);
    }
    return count;
  }

  // True when POPCNT is odd.
  constexpr bool POPPAR() const { return POPCNT() & 1; }

  constexpr int TRAILZ() const {
    auto minus1{AddUnsigned(MASKR(bits))}; // { x-1, carry = x > 0 }
    if (!minus1.carry) {
      return bits; // was zero
    } else {
      // x ^ (x-1) has all bits set at and below original least-order set bit.
      return IEOR(minus1.value).POPCNT() - 1;
    }
  }

  constexpr bool BTEST(int pos) const {
    if (pos < 0 || pos >= bits) {
      return false;
    } else {
      return (LEPart(pos / partBits) >> (pos % partBits)) & 1;
    }
  }

  constexpr Ordering CompareUnsigned(const Integer &y) const {
    for (int j{parts}; j-- > 0;) {
      if (LEPart(j) > y.LEPart(j)) {
        return Ordering::Greater;
      }
      if (LEPart(j) < y.LEPart(j)) {
        return Ordering::Less;
      }
    }
    return Ordering::Equal;
  }

  constexpr bool BGE(const Integer &y) const {
    return CompareUnsigned(y) != Ordering::Less;
  }
  constexpr bool BGT(const Integer &y) const {
    return CompareUnsigned(y) == Ordering::Greater;
  }
  constexpr bool BLE(const Integer &y) const { return !BGT(y); }
  constexpr bool BLT(const Integer &y) const { return !BGE(y); }

  constexpr Ordering CompareSigned(const Integer &y) const {
    bool isNegative{IsNegative()};
    if (isNegative != y.IsNegative()) {
      return isNegative ? Ordering::Less : Ordering::Greater;
    }
    return CompareUnsigned(y);
  }

  constexpr std::uint64_t ToUInt64() const {
    std::uint64_t n{LEPart(0)};
    int filled{partBits};
    for (int j{1}; filled < 64 && j < parts; ++j, filled += partBits) {
      n |= std::uint64_t{LEPart(j)} << filled;
    }
    return n;
  }

  constexpr std::int64_t ToInt64() const {
    std::int64_t signExtended = ToUInt64();
    if constexpr (bits < 64) {
      signExtended |= -(signExtended >> (bits - 1)) << bits;
    }
    return signExtended;
  }

  // Ones'-complement (i.e., C's ~)
  constexpr Integer NOT() const {
    Integer result{nullptr};
    for (int j{0}; j < parts; ++j) {
      result.SetLEPart(j, ~LEPart(j));
    }
    return result;
  }

  // Two's-complement negation (-x = ~x + 1).
  // An overflow flag accompanies the result, and will be true when the
  // operand is the most negative signed number (MASKL(1)).
  constexpr ValueWithOverflow Negate() const {
    Integer result{nullptr};
    Part carry{1};
    for (int j{0}; j + 1 < parts; ++j) {
      Part newCarry{LEPart(j) == 0 && carry};
      result.SetLEPart(j, ~LEPart(j) + carry);
      carry = newCarry;
    }
    Part top{LEPart(parts - 1)};
    result.SetLEPart(parts - 1, ~top + carry);
    bool overflow{top != 0 && result.LEPart(parts - 1) == top};
    return {result, overflow};
  }

  constexpr ValueWithOverflow ABS() const {
    if (IsNegative()) {
      return Negate();
    } else {
      return {*this, false};
    }
  }

  // Shifts the operand left when the count is positive, right when negative.
  // Vacated bit positions are filled with zeroes.
  constexpr Integer ISHFT(int count) const {
    if (count < 0) {
      return SHIFTR(-count);
    } else {
      return SHIFTL(count);
    }
  }

  // Left shift with zero fill.
  constexpr Integer SHIFTL(int count) const {
    if (count <= 0) {
      return *this;
    } else {
      Integer result{nullptr};
      int shiftParts{count / partBits};
      int bitShift{count - partBits * shiftParts};
      int j{parts - 1};
      if (bitShift == 0) {
        for (; j >= shiftParts; --j) {
          result.SetLEPart(j, LEPart(j - shiftParts));
        }
        for (; j >= 0; --j) {
          result.LEPart(j) = 0;
        }
      } else {
        for (; j > shiftParts; --j) {
          result.SetLEPart(j,
              ((LEPart(j - shiftParts) << bitShift) |
                  (LEPart(j - shiftParts - 1) >> (partBits - bitShift))));
        }
        if (j == shiftParts) {
          result.SetLEPart(j, LEPart(0) << bitShift);
          --j;
        }
        for (; j >= 0; --j) {
          result.LEPart(j) = 0;
        }
      }
      return result;
    }
  }

  // Circular shift of a field of least-significant bits.  The least-order
  // "size" bits are shifted circularly in place by "count" positions;
  // the shift is leftward if count is nonnegative, rightward otherwise.
  // Higher-order bits are unchanged.
  constexpr Integer ISHFTC(int count, int size = bits) const {
    if (count == 0 || size <= 0) {
      return *this;
    }
    if (size > bits) {
      size = bits;
    }
    count %= size;
    if (count == 0) {
      return *this;
    }
    int middleBits{size - count}, leastBits{count};
    if (count < 0) {
      middleBits = -count;
      leastBits = size + count;
    }
    if (size == bits) {
      return SHIFTL(leastBits).IOR(SHIFTR(middleBits));
    }
    Integer unchanged{IAND(MASKL(bits - size))};
    Integer middle{IAND(MASKR(middleBits)).SHIFTL(leastBits)};
    Integer least{SHIFTR(middleBits).IAND(MASKR(leastBits))};
    return unchanged.IOR(middle).IOR(least);
  }

  // Double shifts, aka shifts with specific fill.
  constexpr Integer SHIFTLWithFill(const Integer &fill, int count) const {
    if (count <= 0) {
      return *this;
    } else if (count >= 2 * bits) {
      return {};
    } else if (count > bits) {
      return fill.SHIFTL(count - bits);
    } else if (count == bits) {
      return fill;
    } else {
      return SHIFTL(count).IOR(fill.SHIFTR(bits - count));
    }
  }

  constexpr Integer SHIFTRWithFill(const Integer &fill, int count) const {
    if (count <= 0) {
      return *this;
    } else if (count >= 2 * bits) {
      return {};
    } else if (count > bits) {
      return fill.SHIFTR(count - bits);
    } else if (count == bits) {
      return fill;
    } else {
      return SHIFTR(count).IOR(fill.SHIFTL(bits - count));
    }
  }

  constexpr Integer DSHIFTL(const Integer &fill, int count) const {
    // DSHIFTL(I,J) shifts I:J left; the second argument is the right fill.
    return SHIFTLWithFill(fill, count);
  }

  constexpr Integer DSHIFTR(const Integer &value, int count) const {
    // DSHIFTR(I,J) shifts I:J right; the *first* argument is the left fill.
    return value.SHIFTRWithFill(*this, count);
  }

  // Vacated upper bits are filled with zeroes.
  constexpr Integer SHIFTR(int count) const {
    if (count <= 0) {
      return *this;
    } else {
      Integer result{nullptr};
      int shiftParts{count / partBits};
      int bitShift{count - partBits * shiftParts};
      int j{0};
      if (bitShift == 0) {
        for (; j + shiftParts < parts; ++j) {
          result.LEPart(j) = LEPart(j + shiftParts);
        }
        for (; j < parts; ++j) {
          result.LEPart(j) = 0;
        }
      } else {
        for (; j + shiftParts + 1 < parts; ++j) {
          result.SetLEPart(j,
              (LEPart(j + shiftParts) >> bitShift) |
                  (LEPart(j + shiftParts + 1) << (partBits - bitShift)));
        }
        if (j + shiftParts + 1 == parts) {
          result.LEPart(j++) = LEPart(parts - 1) >> bitShift;
        }
        for (; j < parts; ++j) {
          result.LEPart(j) = 0;
        }
      }
      return result;
    }
  }

  // Be advised, an arithmetic (sign-filling) right shift is not
  // the same as a division by a power of two in all cases.
  constexpr Integer SHIFTA(int count) const {
    if (count <= 0) {
      return *this;
    } else if (IsNegative()) {
      return SHIFTR(count).IOR(MASKL(count));
    } else {
      return SHIFTR(count);
    }
  }

  // Clears a single bit.
  constexpr Integer IBCLR(int pos) const {
    if (pos < 0 || pos >= bits) {
      return *this;
    } else {
      Integer result{*this};
      result.LEPart(pos / partBits) &= ~(Part{1} << (pos % partBits));
      return result;
    }
  }

  // Sets a single bit.
  constexpr Integer IBSET(int pos) const {
    if (pos < 0 || pos >= bits) {
      return *this;
    } else {
      Integer result{*this};
      result.LEPart(pos / partBits) |= Part{1} << (pos % partBits);
      return result;
    }
  }

  // Extracts a field.
  constexpr Integer IBITS(int pos, int size) const {
    return SHIFTR(pos).IAND(MASKR(size));
  }

  constexpr Integer IAND(const Integer &y) const {
    Integer result{nullptr};
    for (int j{0}; j < parts; ++j) {
      result.LEPart(j) = LEPart(j) & y.LEPart(j);
    }
    return result;
  }

  constexpr Integer IOR(const Integer &y) const {
    Integer result{nullptr};
    for (int j{0}; j < parts; ++j) {
      result.LEPart(j) = LEPart(j) | y.LEPart(j);
    }
    return result;
  }

  constexpr Integer IEOR(const Integer &y) const {
    Integer result{nullptr};
    for (int j{0}; j < parts; ++j) {
      result.LEPart(j) = LEPart(j) ^ y.LEPart(j);
    }
    return result;
  }

  constexpr Integer MERGE_BITS(const Integer &y, const Integer &mask) const {
    return IAND(mask).IOR(y.IAND(mask.NOT()));
  }

  constexpr Integer MAX(const Integer &y) const {
    if (CompareSigned(y) == Ordering::Less) {
      return y;
    } else {
      return *this;
    }
  }

  constexpr Integer MIN(const Integer &y) const {
    if (CompareSigned(y) == Ordering::Less) {
      return *this;
    } else {
      return y;
    }
  }

  // Unsigned addition with carry.
  constexpr ValueWithCarry AddUnsigned(
      const Integer &y, bool carryIn = false) const {
    Integer sum{nullptr};
    BigPart carry{carryIn};
    for (int j{0}; j + 1 < parts; ++j) {
      carry += LEPart(j);
      carry += y.LEPart(j);
      sum.SetLEPart(j, carry);
      carry >>= partBits;
    }
    carry += LEPart(parts - 1);
    carry += y.LEPart(parts - 1);
    sum.SetLEPart(parts - 1, carry);
    return {sum, carry > topPartMask};
  }

  constexpr ValueWithOverflow AddSigned(const Integer &y) const {
    bool isNegative{IsNegative()};
    bool sameSign{isNegative == y.IsNegative()};
    ValueWithCarry sum{AddUnsigned(y)};
    bool overflow{sameSign && sum.value.IsNegative() != isNegative};
    return {sum.value, overflow};
  }

  constexpr ValueWithOverflow SubtractSigned(const Integer &y) const {
    bool isNegative{IsNegative()};
    bool sameSign{isNegative == y.IsNegative()};
    ValueWithCarry diff{AddUnsigned(y.Negate().value)};
    bool overflow{!sameSign && diff.value.IsNegative() != isNegative};
    return {diff.value, overflow};
  }

  // MAX(X-Y, 0)
  constexpr Integer DIM(const Integer &y) const {
    if (CompareSigned(y) != Ordering::Greater) {
      return {};
    } else {
      return SubtractSigned(y).value;
    }
  }

  constexpr ValueWithOverflow SIGN(bool toNegative) const {
    if (toNegative == IsNegative()) {
      return {*this, false};
    } else if (toNegative) {
      return Negate();
    } else {
      return ABS();
    }
  }

  constexpr ValueWithOverflow SIGN(const Integer &sign) const {
    return SIGN(sign.IsNegative());
  }

  constexpr Product MultiplyUnsigned(const Integer &y) const {
    Part product[2 * parts]{}; // little-endian full product
    for (int j{0}; j < parts; ++j) {
      if (Part xpart{LEPart(j)}) {
        for (int k{0}; k < parts; ++k) {
          if (Part ypart{y.LEPart(k)}) {
            BigPart xy{xpart};
            xy *= ypart;
#if defined __GNUC__ && __GNUC__ < 8
            // && to < (2 * parts) was added to avoid GCC < 8 build failure on
            // -Werror=array-bounds. This can be removed if -Werror is disable.
            for (int to{j + k}; xy != 0 && to < (2 * parts); ++to) {
#else
            for (int to{j + k}; xy != 0; ++to) {
#endif
              xy += product[to];
              product[to] = xy & partMask;
              xy >>= partBits;
            }
          }
        }
      }
    }
    Integer upper{nullptr}, lower{nullptr};
    for (int j{0}; j < parts; ++j) {
      lower.LEPart(j) = product[j];
      upper.LEPart(j) = product[j + parts];
    }
    if constexpr (topPartBits < partBits) {
      upper = upper.SHIFTL(partBits - topPartBits);
      upper.LEPart(0) |= lower.LEPart(parts - 1) >> topPartBits;
      lower.LEPart(parts - 1) &= topPartMask;
    }
    return {upper, lower};
  }

  constexpr Product MultiplySigned(const Integer &y) const {
    bool yIsNegative{y.IsNegative()};
    Integer absy{y};
    if (yIsNegative) {
      absy = y.Negate().value;
    }
    bool isNegative{IsNegative()};
    Integer absx{*this};
    if (isNegative) {
      absx = Negate().value;
    }
    Product product{absx.MultiplyUnsigned(absy)};
    if (isNegative != yIsNegative) {
      product.lower = product.lower.NOT();
      product.upper = product.upper.NOT();
      Integer one{1};
      auto incremented{product.lower.AddUnsigned(one)};
      product.lower = incremented.value;
      if (incremented.carry) {
        product.upper = product.upper.AddUnsigned(one).value;
      }
    }
    return product;
  }

  constexpr QuotientWithRemainder DivideUnsigned(const Integer &divisor) const {
    if (divisor.IsZero()) {
      return {MASKR(bits), Integer{}, true, false}; // overflow to max value
    }
    int bitsDone{LEADZ()};
    Integer top{SHIFTL(bitsDone)};
    Integer quotient, remainder;
    for (; bitsDone < bits; ++bitsDone) {
      auto doubledTop{top.AddUnsigned(top)};
      top = doubledTop.value;
      remainder = remainder.AddUnsigned(remainder, doubledTop.carry).value;
      bool nextBit{remainder.CompareUnsigned(divisor) != Ordering::Less};
      quotient = quotient.AddUnsigned(quotient, nextBit).value;
      if (nextBit) {
        remainder = remainder.SubtractSigned(divisor).value;
      }
    }
    return {quotient, remainder, false, false};
  }

  // A nonzero remainder has the sign of the dividend, i.e., it computes
  // the MOD intrinsic (X-INT(X/Y)*Y), not MODULO (which is below).
  // 8/5 = 1r3;  -8/5 = -1r-3;  8/-5 = -1r3;  -8/-5 = 1r-3
  constexpr QuotientWithRemainder DivideSigned(Integer divisor) const {
    bool dividendIsNegative{IsNegative()};
    bool negateQuotient{dividendIsNegative};
    Ordering divisorOrdering{divisor.CompareToZeroSigned()};
    if (divisorOrdering == Ordering::Less) {
      negateQuotient = !negateQuotient;
      auto negated{divisor.Negate()};
      if (negated.overflow) {
        // divisor was (and is) the most negative number
        if (CompareUnsigned(divisor) == Ordering::Equal) {
          return {MASKR(1), Integer{}, false, bits <= 1};
        } else {
          return {Integer{}, *this, false, false};
        }
      }
      divisor = negated.value;
    } else if (divisorOrdering == Ordering::Equal) {
      // division by zero
      if (dividendIsNegative) {
        return {MASKL(1), Integer{}, true, false};
      } else {
        return {MASKR(bits - 1), Integer{}, true, false};
      }
    }
    Integer dividend{*this};
    if (dividendIsNegative) {
      auto negated{Negate()};
      if (negated.overflow) {
        // Dividend was (and remains) the most negative number.
        // See whether the original divisor was -1 (if so, it's 1 now).
        if (divisorOrdering == Ordering::Less &&
            divisor.CompareUnsigned(Integer{1}) == Ordering::Equal) {
          // most negative number / -1 is the sole overflow case
          return {*this, Integer{}, false, true};
        }
      } else {
        dividend = negated.value;
      }
    }
    // Overflow is not possible, and both the dividend and divisor
    // are now positive.
    QuotientWithRemainder result{dividend.DivideUnsigned(divisor)};
    if (negateQuotient) {
      result.quotient = result.quotient.Negate().value;
    }
    if (dividendIsNegative) {
      result.remainder = result.remainder.Negate().value;
    }
    return result;
  }

  // Result has the sign of the divisor argument.
  // 8 mod 5 = 3;  -8 mod 5 = 2;  8 mod -5 = -2;  -8 mod -5 = -3
  constexpr ValueWithOverflow MODULO(const Integer &divisor) const {
    bool negativeDivisor{divisor.IsNegative()};
    bool distinctSigns{IsNegative() != negativeDivisor};
    QuotientWithRemainder divided{DivideSigned(divisor)};
    if (distinctSigns && !divided.remainder.IsZero()) {
      return {divided.remainder.AddUnsigned(divisor).value, divided.overflow};
    } else {
      return {divided.remainder, divided.overflow};
    }
  }

  constexpr PowerWithErrors Power(const Integer &exponent) const {
    PowerWithErrors result{1, false, false, false};
    if (exponent.IsZero()) {
      // x**0 -> 1, including the case 0**0, which is not defined specifically
      // in F'18 afaict; however, other Fortrans tested all produce 1, not 0,
      // apart from nagfor, which stops with an error at runtime.
      // Ada, APL, C's pow(), Haskell, Julia, MATLAB, and R all produce 1 too.
      // F'77 explicitly states that 0**0 is mathematically undefined and
      // therefore prohibited.
      result.zeroToZero = IsZero();
    } else if (exponent.IsNegative()) {
      if (IsZero()) {
        result.divisionByZero = true;
        result.power = MASKR(bits - 1);
      } else if (CompareSigned(Integer{1}) == Ordering::Equal) {
        result.power = *this; // 1**x -> 1
      } else if (CompareSigned(Integer{-1}) == Ordering::Equal) {
        if (exponent.BTEST(0)) {
          result.power = *this; // (-1)**x -> -1 if x is odd
        }
      } else {
        result.power.Clear(); // j**k -> 0 if |j| > 1 and k < 0
      }
    } else {
      Integer shifted{*this};
      Integer pow{exponent};
      int nbits{bits - pow.LEADZ()};
      for (int j{0}; j < nbits; ++j) {
        if (pow.BTEST(j)) {
          Product product{result.power.MultiplySigned(shifted)};
          result.power = product.lower;
          result.overflow |= product.SignedMultiplicationOverflowed();
        }
        if (j + 1 < nbits) {
          Product squared{shifted.MultiplySigned(shifted)};
          result.overflow |= squared.SignedMultiplicationOverflowed();
          shifted = squared.lower;
        }
      }
    }
    return result;
  }

private:
  // A private constructor, selected by the use of nullptr,
  // that is used by member functions when it would be a waste
  // of time to initialize parts_[].
  constexpr Integer(std::nullptr_t) {}

  // Accesses parts in little-endian order.
  constexpr const Part &LEPart(int part) const {
    if constexpr (littleEndian) {
      return part_[part];
    } else {
      return part_[parts - 1 - part];
    }
  }

  constexpr Part &LEPart(int part) {
    if constexpr (littleEndian) {
      return part_[part];
    } else {
      return part_[parts - 1 - part];
    }
  }

  constexpr void SetLEPart(int part, Part x) {
    LEPart(part) = x & PartMask(part);
  }

  static constexpr Part PartMask(int part) {
    return part == parts - 1 ? topPartMask : partMask;
  }

  constexpr void Clear() {
    for (int j{0}; j < parts; ++j) {
      part_[j] = 0;
    }
  }

  Part part_[parts]{};
};

extern template class Integer<8>;
extern template class Integer<16>;
extern template class Integer<32>;
extern template class Integer<64>;
extern template class Integer<80>;
extern template class Integer<128>;
} // namespace Fortran::evaluate::value
#endif // FORTRAN_EVALUATE_INTEGER_H_
