//===-- include/flang/Evaluate/integer-value.h ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_EVALUATE_INTEGER_VALUE_H_
#define FORTRAN_EVALUATE_INTEGER_VALUE_H_

#include "flang/Common/uint128.h"
#include "flang/Evaluate/common.h"
#include "flang/Evaluate/object-sizes.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <type_traits>

// Some environments, viz. glibc 2.17 and *BSD, allow the macro HUGE
// to leak out of <math.h>.
#undef HUGE

namespace Fortran::evaluate::value {
class IntegerValueImpl;

/// A two's-complement integer with dynamic bitwidth.
///
/// The bitwidth is dynamic, but only a predefined set of Fortran kinds are
/// allowed. It is also kind-aware, i.e. knows which INTEGER kind it currently
/// represents.
///
/// The implementation is hidden from this header using a pImpl-like idiom.
class IntegerValue {
  friend class RealValueImpl;

public:
  struct ValueWithOverflow;
  struct ValueWithCarry;
  struct Product;
  struct QuotientWithRemainder;
  struct PowerWithErrors;

  IntegerValue();
  ~IntegerValue();
  IntegerValue(const IntegerValue &);
  IntegerValue(IntegerValue &&);
  IntegerValue &operator=(const IntegerValue &);
  IntegerValue &operator=(IntegerValue &&);

  IntegerValue(int kind, const IntegerValue &x) : IntegerValue(x) {
    CHECK(x.kind() == kind);
  }
  IntegerValue(int kind, IntegerValue &&x) : IntegerValue(std::move(x)) {
    CHECK(x.kind() == kind);
  }

  // Fortran::common::int128_t/uint128_t are 128-bit values -- either the
  // host's native __int128/unsigned __int128, or the portable
  // Fortran::common::Int128<> fallback when there is no native type -- and
  // are handled by the dedicated branch below rather than by the general
  // integral case, since some standard libraries don't consider native
  // __int128 types to satisfy std::is_integral_v, and the portable fallback
  // is a class type that never does.
  template <typename INT,
      typename = std::enable_if_t<std::numeric_limits<INT>::is_integer>>
  IntegerValue(int kind, INT v) {
    if constexpr (sizeof(INT) > 8) {
      static_assert(sizeof(INT) == 16);
      ConstructFromIntegral(kind, static_cast<Fortran::common::uint128_t>(v));
    } else if constexpr (std::is_signed_v<INT>) {
      ConstructFromIntegral(
          kind, static_cast<uint64_t>(static_cast<int64_t>(v)), true);
    } else {
      ConstructFromIntegral(kind, static_cast<uint64_t>(v), false);
    }
  }

  /// Creates an integer with value 0 of a given kind. This is different from
  /// the default-ctor which creates a "monostate" that represents 0 of unknown
  /// kind.
  static IntegerValue Zero(int kind);

  void print(llvm::raw_ostream &os) const;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump() const;
#endif

  /// Whether this object represents a default-initialized value (zero) of
  /// not-yet-known kind.
  bool IsMonostate() const;

  /// The kind of the value currently stored.
  int kind() const;

  int bits() const { return bits(kind()); }
  static constexpr int bits(int kind) { return bytesStored(kind) * 8; }

  /// Number of bytes accessed by FromRawBytes/StoreRawBytes
  std::size_t bytesStored() const { return bytesStored(kind()); }
  static constexpr std::size_t bytesStored(int kind) {
    switch (kind) {
    case 3:
      return 2;
    case 10:
      return 16;
    default:
      return kind;
    }
  }

  bool operator<(const IntegerValue &y) const {
    return CompareSigned(y) == Ordering::Less;
  }
  bool operator<=(const IntegerValue &y) const { return !(y < *this); }
  bool operator==(const IntegerValue &y) const;
  bool operator!=(const IntegerValue &y) const { return !(*this == y); }
  bool operator>=(const IntegerValue &y) const { return !(*this < y); }
  bool operator>(const IntegerValue &y) const { return y < *this; }

  /// Left-justified mask (e.g., MASKL(1) has only its sign bit set)
  static IntegerValue MASKL(int kind, int places);

  /// Right-justified mask (e.g., MASKR(1) == 1, MASKR(2) == 3, &c.)
  static IntegerValue MASKR(int kind, int places);

  static ValueWithOverflow Read(
      int kind, const char *&pp, int base, bool isSigned);

  /// ZExt or Trunc
  static ValueWithOverflow ConvertUnsigned(
      int toKind, const IntegerValue &from);

  /// SExt or Trunc
  static ValueWithOverflow ConvertSigned(int toKind, const IntegerValue &from);

  std::string UnsignedDecimal() const;

  std::string SignedDecimal() const;

  /// Omits a leading "0x".
  std::string Hexadecimal() const;

  static constexpr int DIGITS(int kind) {
    // don't count the sign bit
    return bits(kind) - 1;
  }

  static IntegerValue HUGE(int kind);

  static IntegerValue Least(int kind);

  static int RANGE(int kind);

  static int UnsignedRANGE(int kind);

  bool IsZero() const;

  bool IsNegative() const;

  Ordering CompareToZeroSigned() const;

  /// Count the number of contiguous most-significant bit positions
  /// that are clear.
  int LEADZ() const;

  /// Count the number of bit positions that are set.
  int POPCNT() const;

  /// True when POPCNT is odd.
  bool POPPAR() const;

  int TRAILZ() const;

  bool BTEST(int pos) const;

  Ordering CompareUnsigned(const IntegerValue &y) const;

  Ordering CompareSigned(const IntegerValue &y) const;

  bool BGE(const IntegerValue &y) const {
    return CompareUnsigned(y) != Ordering::Less;
  }
  bool BGT(const IntegerValue &y) const {
    return CompareUnsigned(y) == Ordering::Greater;
  }
  bool BLE(const IntegerValue &y) const { return !BGT(y); }
  bool BLT(const IntegerValue &y) const { return !BGE(y); }

  std::uint64_t ToUInt64() const;

  std::int64_t ToInt64() const;

  Fortran::common::uint128_t ToUInt128() const;

  Fortran::common::int128_t ToInt128() const;

  template <typename INT,
      typename = std::enable_if_t<std::is_signed_v<INT> ||
          std::is_same_v<INT, Fortran::common::int128_t>>>
  INT ToSInt() const {
    if constexpr (std::is_same_v<INT, Fortran::common::int128_t>) {
      return ToInt128();
    } else {
      return ToInt64();
    }
  }

  template <typename INT,
      typename = std::enable_if_t<std::is_unsigned_v<INT> ||
          std::is_same_v<INT, Fortran::common::uint128_t>>>
  INT ToUInt() const {
    if constexpr (std::is_same_v<INT, Fortran::common::uint128_t>) {
      return ToUInt128();
    } else {
      return ToUInt64();
    }
  }

  /// Ones'-complement (i.e., C's ~)
  IntegerValue NOT() const;

  /// Two's-complement negation (-x = ~x + 1).
  /// An overflow flag accompanies the result, and will be true when the
  /// operand is the most negative signed number (MASKL(1)).
  ValueWithOverflow Negate() const;

  ValueWithOverflow ABS() const;

  /// Shifts the operand left when the count is positive, right when negative.
  /// Vacated bit positions are filled with zeroes.
  IntegerValue ISHFT(int count) const {
    return count < 0 ? SHIFTR(-count) : SHIFTL(count);
  }

  /// Left shift with zero fill.
  IntegerValue SHIFTL(int count) const;

  /// Circular shift of a field of least-significant bits.  The least-order
  /// "size" bits are shifted circularly in place by "count" positions;
  /// the shift is leftward if count is nonnegative, rightward otherwise.
  /// Higher-order bits are unchanged.
  IntegerValue ISHFTC(int count, int size) const;
  IntegerValue ISHFTC(int count) const;

  /// DSHIFTL(I,J) shifts I:J left; the second argument is the right fill.
  IntegerValue DSHIFTL(const IntegerValue &fill, int count) const;

  /// DSHIFTR(I,J) shifts I:J right; the *first* argument is the left fill.
  IntegerValue DSHIFTR(const IntegerValue &v2, int count) const;

  /// Vacated upper bits are filled with zeroes.
  IntegerValue SHIFTR(int count) const;

  /// Be advised, an arithmetic (sign-filling) right shift is not
  /// the same as a division by a power of two in all cases.
  IntegerValue SHIFTA(int count) const;

  /// Clears a single bit.
  IntegerValue IBCLR(int pos) const;

  /// Sets a single bit.
  IntegerValue IBSET(int pos) const;

  /// Extracts a field.
  IntegerValue IBITS(int pos, int size) const;

  IntegerValue IAND(const IntegerValue &y) const;

  IntegerValue IOR(const IntegerValue &y) const;

  IntegerValue IEOR(const IntegerValue &y) const;

  IntegerValue MERGE_BITS(
      const IntegerValue &y, const IntegerValue &mask) const;

  IntegerValue MAX(const IntegerValue &y) const {
    return CompareSigned(y) == Ordering::Less ? y : *this;
  }

  IntegerValue MIN(const IntegerValue &y) const {
    return CompareSigned(y) == Ordering::Less ? *this : y;
  }

  ValueWithCarry AddUnsigned(const IntegerValue &y, bool carryIn = false) const;

  ValueWithOverflow AddSigned(const IntegerValue &y) const;

  ValueWithOverflow SubtractSigned(const IntegerValue &y) const;

  /// DIM(X,Y)=MAX(X-Y, 0)
  ValueWithOverflow DIM(const IntegerValue &y) const;

  ValueWithOverflow SIGN(const IntegerValue &sign) const;

  Product MultiplyUnsigned(const IntegerValue &y) const;

  Product MultiplySigned(const IntegerValue &y) const;

  QuotientWithRemainder DivideUnsigned(const IntegerValue &y) const;

  /// A nonzero remainder has the sign of the dividend, i.e., it computes
  /// the MOD intrinsic (X-INT(X/Y)*Y), not MODULO (which is below).
  /// 8/5 = 1r3;  -8/5 = -1r-3;  8/-5 = -1r3;  -8/-5 = 1r-3
  QuotientWithRemainder DivideSigned(const IntegerValue &y) const;

  /// Result has the sign of the divisor argument.
  /// 8 mod 5 = 3;  -8 mod 5 = 2;  8 mod -5 = -2;  -8 mod -5 = -3
  ValueWithOverflow MODULO(const IntegerValue &y) const;

  PowerWithErrors Power(const IntegerValue &e) const;

  static IntegerValue FromRawBytes(
      int kind, const void *raw, std::size_t expectedSize);
  void StoreRawBytes(void *dst, size_t size, bool *changed = nullptr) const;

private:
  void ConstructFromIntegral(int kind, std::uint64_t n, bool isSigned);
  void ConstructFromIntegral(int kind, Fortran::common::uint128_t n);

  static IntegerValue FromImpl(const IntegerValueImpl &x);
  static IntegerValue FromImpl(IntegerValueImpl &&x);

  IntegerValueImpl &impl() {
    return *reinterpret_cast<IntegerValueImpl *>(this);
  }
  const IntegerValueImpl &impl() const {
    return *reinterpret_cast<const IntegerValueImpl *>(this);
  }

  [[maybe_unused]] alignas(
      detail::kIntegerObjectAlign) char opaque_[detail::kIntegerObjectSize];
};

struct IntegerValue::ValueWithOverflow {
  IntegerValue value;
  bool overflow{false};
};

struct IntegerValue::ValueWithCarry {
  IntegerValue value;
  bool carry{false};
};

struct IntegerValue::Product {
  IntegerValue upper, lower;
  bool SignedMultiplicationOverflowed() const { return overflow; }
  bool overflow{false};
};

struct IntegerValue::QuotientWithRemainder {
  IntegerValue quotient, remainder;
  bool divisionByZero{false}, overflow{false};
};

struct IntegerValue::PowerWithErrors {
  IntegerValue power;
  bool divisionByZero{false}, overflow{false}, zeroToZero{false};
};

} // namespace Fortran::evaluate::value

namespace llvm {
/// For pretty printing in GTest
inline raw_ostream &operator<<(
    raw_ostream &os, const Fortran::evaluate::value::IntegerValue &v) {
  v.print(os);
  return os;
}
} // namespace llvm

#endif // FORTRAN_EVALUATE_INTEGER_VALUE_H_
