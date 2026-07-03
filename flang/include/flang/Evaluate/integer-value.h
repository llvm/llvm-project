//===-- include/flang/Evaluate/integer-value.h ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_EVALUATE_INTEGER_VALUE_H_
#define FORTRAN_EVALUATE_INTEGER_VALUE_H_

// Emulates binary integers of an arbitrary (but fixed) bit size for use
// when the host C++ environment does not support that size or when the
// full suite of Fortran's integer intrinsic scalar functions are needed.
// The data model is typeless, so signed* and unsigned operations
// are distinguished from each other with distinct member function interfaces.
// (*"Signed" here means two's-complement, just to be clear.  Ones'-complement
// and signed-magnitude encodings appear to be extinct in 2018.)

#include "flang/Evaluate/common.h"
#include "llvm/ADT/APInt.h"
#include <cstdint>
#include <type_traits>

// Some environments, viz. glibc 2.17 and *BSD, allow the macro HUGE
// to leak out of <math.h>.
#undef HUGE

namespace Fortran::evaluate::value {

// Detects the runtime-kind integer facade (which exposes toAPInt()) vs. a raw
// fixed-width value::Integer.
template <typename, typename = void> struct HasWord : std::false_type {};
template <typename T>
struct HasWord<T, std::void_t<decltype(std::declval<const T &>().toAPInt())>>
    : std::true_type {};

// ----------------------------------------------------------------------------
// IntegerValue: runtime-kind two's-complement integer (also used for UNSIGNED).
// ----------------------------------------------------------------------------
class IntegerValue {
public:
  // The value is held as a single arbitrary-precision integer; its bit width is
  // the runtime kind's width (8/16/32/64/80/128).  A zero-width APInt
  // (APInt::getZeroWidth()) models the former monostate: a default-initialized
  // value of as-yet-unknown width.
  using Storage = llvm::APInt;

  // These result aggregates hold IntegerValue members by value, so they cannot
  // be defined until IntegerValue itself is a complete type.  Forward-declare
  // them here (member declarations may return an incomplete type) and define
  // them immediately after the class below.
  struct ValueWithOverflow;
  struct ValueWithCarry;
  struct Product;
  struct QuotientWithRemainder;
  struct PowerWithErrors;

  // Constructors
  IntegerValue() {}
  IntegerValue(const IntegerValue &) = default;
  IntegerValue(IntegerValue &&) = default;
  IntegerValue &operator=(const IntegerValue &) = default;
  IntegerValue &operator=(IntegerValue &&) = default;

  // PAPAYA: Replace with IntegerValue(std::uint64_t, int kind, bool isSigned);
  // Don't rely on the signedness of the type; int128_t also not supported
  template <typename INT, typename = std::enable_if_t<std::is_integral_v<INT>>>
  IntegerValue(INT n, int kind) {
    // PAPAYA:   assert(n <= UINT64_MAX);
    // The runtime kind is carried by the APInt's bit width.  A signed host
    // value is sign-extended, an unsigned one zero-extended, then silently
    // truncated to the requested width (matching value::Integer's integral
    // constructor; implicitTrunc avoids APInt's "does not fit" assertion).
    storage_ = llvm::APInt(BitsForKind(kind), static_cast<std::uint64_t>(n),
        std::is_signed_v<INT>,
        /*implicitTrunc=*/true);
  }

  // Comparison operators
  bool operator==(const IntegerValue &y) const;
  bool operator!=(const IntegerValue &y) const { return !(*this == y); }
  bool operator<(const IntegerValue &y) const {
    return CompareSigned(y) == Ordering::Less;
  }
  bool operator>(const IntegerValue &y) const { return y < *this; }
  bool operator<=(const IntegerValue &y) const { return !(y < *this); }
  bool operator>=(const IntegerValue &y) const { return !(*this < y); }

  // Left-justified mask (e.g., MASKL(1) has only its sign bit set)
  static IntegerValue MASKL(int places, int kind);
  // Right-justified mask (e.g., MASKR(1) == 1, MASKR(2) == 3, &c.)
  static IntegerValue MASKR(int places, int kind);
  static IntegerValue HUGE(int kind);
  static IntegerValue Least(int kind);

  // Decimal exponent range, formerly a compile-time constant; now selected by
  // the runtime kind (1/2/4/8/16).
  static int RANGE(int kind);
  static int UnsignedRANGE(int kind);
  // Binary digits excluding the sign bit (== bits - 1).
  static int DIGITS(int kind) { return 8 * kind - 1; }

  // Runtime kind / width accessors
  int kind() const;
  int bits() const;
  // A zero-width APInt is never a real value (every Fortran integer kind has a
  // positive width), so it uniquely marks the default-initialized state.
  bool IsMonostate() const { return storage_.getBitWidth() == 0; }
  bool IsZero() const;
  bool IsNegative() const;
  bool StoreRawBytes(void *to, int kind) const;
  static IntegerValue FromRawBytes(const void *raw, int kind);
  static IntegerValue Zero(int kind);

  // Raw access to the underlying arbitrary-precision value (precondition: not
  // monostate).  Used by RealValue/ComplexValue to bridge to llvm::APFloat.
  const llvm::APInt &toAPInt() const { return ap(); }
  static IntegerValue FromAPInt(const llvm::APInt &i) { return Wrap(i); }

  std::uint64_t ToUInt64() const;
  std::int64_t ToInt64() const;
  std::uint64_t ToUInt() const;

  template <typename SINT = std::int64_t> SINT ToSInt() const {
    if (IsMonostate()) {
      return 0;
    }
    return static_cast<SINT>(ap().getRawData()[0]);
  }

  // Signed/unsigned comparisons
  Ordering CompareSigned(const IntegerValue &y) const;
  Ordering CompareUnsigned(const IntegerValue &y) const;
  Ordering CompareToZeroSigned() const;
  bool BGE(const IntegerValue &y) const;
  bool BGT(const IntegerValue &y) const;
  bool BLE(const IntegerValue &y) const;
  bool BLT(const IntegerValue &y) const;

  // Arithmetic
  ValueWithOverflow Negate() const;
  ValueWithOverflow ABS() const;

  ValueWithCarry AddUnsigned(const IntegerValue &y, bool carryIn = false) const;
  ValueWithOverflow AddSigned(const IntegerValue &y) const;
  ValueWithOverflow SubtractSigned(const IntegerValue &y) const;
  ValueWithOverflow AddUnsignedToOverflow(const IntegerValue &y) const;
  ValueWithOverflow DIM(const IntegerValue &y) const;
  ValueWithOverflow SIGN(const IntegerValue &sign) const;

  Product MultiplySigned(const IntegerValue &y) const;
  Product MultiplyUnsigned(const IntegerValue &y) const;
  QuotientWithRemainder DivideSigned(const IntegerValue &y) const;
  QuotientWithRemainder DivideUnsigned(const IntegerValue &y) const;
  ValueWithOverflow MODULO(const IntegerValue &y) const;
  PowerWithErrors Power(const IntegerValue &e) const;

  // Bitwise operations
  IntegerValue NOT() const;
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

  // Shift operations
  IntegerValue ISHFT(int count) const;
  IntegerValue SHIFTL(int count) const;
  IntegerValue SHIFTR(int count) const;
  IntegerValue SHIFTA(int count) const;
  IntegerValue ISHFTC(int count, int size = 0) const;
  IntegerValue IBITS(int pos, int size) const;
  IntegerValue IBSET(int pos) const;
  IntegerValue IBCLR(int pos) const;
  IntegerValue DSHIFTL(const IntegerValue &fill, int count) const;
  IntegerValue DSHIFTR(const IntegerValue &v2, int count) const;
  bool BTEST(int pos) const;
  int LEADZ() const;
  int TRAILZ() const;
  int POPCNT() const;
  bool POPPAR() const;

  // Returns this value re-interpreted with a different kind (sign-preserving).
  IntegerValue ConvertToKind(int kind) const;

  // The destination kind is no longer a compile-time property, so the target
  // width (in bits, e.g. 8 * TO::kind) is supplied by the caller.
  // Definitions appear after the class, where ValueWithOverflow is complete.
  static ValueWithOverflow ConvertSigned(const IntegerValue &from, int toBits);

  static ValueWithOverflow ConvertUnsigned(
      const IntegerValue &from, int toBits);

  // The result kind is no longer a compile-time property, so the caller
  // supplies the target width (in bits, e.g. 8 * T::kind).  The literal is
  // read directly into that fixed-width alternative so both the stored kind
  // and the overflow flag are correct for the target kind.
  static ValueWithOverflow Read(
      const char *&pp, int base, bool isSigned, int toBits);

  // Formatting
  std::string SignedDecimal() const;
  std::string UnsignedDecimal() const;
  std::string Hexadecimal() const;

private:
  // Wraps an APInt result back into the facade.
  static IntegerValue Wrap(const llvm::APInt &i) {
    IntegerValue r;
    r.storage_ = i;
    return r;
  }

  // The active arbitrary-precision value (precondition: not monostate).
  const llvm::APInt &ap() const { return storage_; }
  unsigned width() const { return ap().getBitWidth(); }

  // y converted (sign-preserving) to this value's width, so that binary
  // operations operate on operands of equal width.  A monostate operand is
  // treated as a zero of this width.
  llvm::APInt coerce(const IntegerValue &y) const {
    if (y.IsMonostate()) {
      return llvm::APInt(width(), 0);
    }
    return y.ap().sextOrTrunc(width());
  }

  // Maps a Fortran integer kind (1/2/4/8/10/16) to its bit width.
  static unsigned BitsForKind(int kind);

  // Double shifts with explicit fill, used by DSHIFTL/DSHIFTR.
  IntegerValue ShiftLWithFill(const IntegerValue &fill, int count) const;
  IntegerValue ShiftRWithFill(const IntegerValue &fill, int count) const;

  // Core width conversion on an APInt source.
  // Definitions appear after the class, where ValueWithOverflow is complete.
  static ValueWithOverflow ConvertAP(
      const llvm::APInt &src, int toBits, bool isSigned);

public: // PAPAYA: make private
  Storage storage_{llvm::APInt::getZeroWidth()};
};

// Out-of-class definitions of the result aggregates (IntegerValue is now a
// complete type, so it may appear as a by-value member).
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

// ConvertAP is defined out-of-line in integer-value.cpp (now that
// ValueWithOverflow is complete its declaration in the private section is
// sufficient here).

} // namespace Fortran::evaluate::value
#endif // FORTRAN_EVALUATE_INTEGER_VALUE_H_
