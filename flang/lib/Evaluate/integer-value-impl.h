//===-- lib/Evaluate/integer-value-impl.h ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_EVALUATE_INTEGER_VALUE_IMPL_H_
#define FORTRAN_EVALUATE_INTEGER_VALUE_IMPL_H_

#include "flang/Evaluate/integer.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>
#include <variant>

// Some environments, viz. glibc 2.17 and *BSD, allow the macro HUGE
// to leak out of <math.h>.
#undef HUGE

namespace Fortran::evaluate::value {

class IntegerValueImpl {
public:
  // Per-KIND fixed-width backing formats.  I80 (X87IntegerContainer) is not
  // itself a Fortran INTEGER kind, but used as REAL(10) storage. While
  // RealValue has its own RealValueImpl, IntegerValue still needs to able to
  // hold it with conversions such as RealValue::IntegerValue().
  using I8 = Integer<8>;
  using I16 = Integer<16>;
  using I32 = Integer<32>;
  using I64 = Integer<64>;
  using I80 = X87IntegerContainer;
  using I128 = Integer<128>;
  using Storage = std::variant<std::monostate, I8, I16, I32, I64, I80, I128>;

  struct ValueWithOverflow;
  struct ValueWithCarry;
  struct Product;
  struct QuotientWithRemainder;
  struct PowerWithErrors;

  // rule-of-five
  ~IntegerValueImpl() = default;
  IntegerValueImpl(const IntegerValueImpl &) = default;
  IntegerValueImpl(IntegerValueImpl &&) = default;
  IntegerValueImpl &operator=(const IntegerValueImpl &) = default;
  IntegerValueImpl &operator=(IntegerValueImpl &&) = default;

  IntegerValueImpl() = default;
  IntegerValueImpl(int kind, const IntegerValueImpl &x) : IntegerValueImpl(x) {
    CHECK(x.kind() == kind);
  }

  static IntegerValueImpl Zero(int kind);

  IntegerValueImpl(int kind, uint64_t v, bool isSigned) {
    withWordProto(kind, [=](auto wordProto) {
      using T = decltype(wordProto);
      storage_ = isSigned ? T{static_cast<int64_t>(v)} : T{v};
    });
  }

  IntegerValueImpl(int kind, Fortran::common::uint128_t v) {
    withWordProto(kind, [=](auto wordProto) {
      using T = decltype(wordProto);
      std::uint64_t lo{static_cast<std::uint64_t>(v)};
      std::uint64_t hi{static_cast<std::uint64_t>(v >> 64)};
      storage_ = T{lo}.IOR(T{hi}.SHIFTL(64));
    });
  }

  template <typename T> static IntegerValueImpl FromWord(const T &n) {
    IntegerValueImpl v;
    v.storage_ = n;
    return v;
  }

  static IntegerValueImpl FromRawBytes(
      int kind, const void *raw, std::size_t expectedSize);

  void print(llvm::raw_ostream &os) const;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump() const;
#endif

  bool IsMonostate() const { return storage_.index() == 0; }
  int kind() const;

  int bits() const;

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

  bool IsZero() const;

  // Comparison operators
  bool operator<(const IntegerValueImpl &y) const {
    return CompareSigned(y) == Ordering::Less;
  }
  bool operator<=(const IntegerValueImpl &y) const { return !(y < *this); }
  bool operator==(const IntegerValueImpl &y) const;
  bool operator!=(const IntegerValueImpl &y) const { return !(*this == y); }
  bool operator>=(const IntegerValueImpl &y) const { return !(*this < y); }
  bool operator>(const IntegerValueImpl &y) const { return y < *this; }

  /// Left-justified mask (e.g., MASKL(1) has only its sign bit set)
  static IntegerValueImpl MASKL(int kind, int places);
  /// Right-justified mask (e.g., MASKR(1) == 1, MASKR(2) == 3, &c.)
  static IntegerValueImpl MASKR(int kind, int places);
  static IntegerValueImpl HUGE(int kind);
  static IntegerValueImpl Least(int kind);

  bool IsNegative() const;

  std::uint64_t ToUInt64() const;
  std::int64_t ToInt64() const;

  Fortran::common::uint128_t ToUInt128() const;
  Fortran::common::int128_t ToInt128() const;

  // Signed/unsigned comparisons
  Ordering CompareSigned(const IntegerValueImpl &y) const;
  Ordering CompareUnsigned(const IntegerValueImpl &y) const;
  Ordering CompareToZeroSigned() const;

  // Arithmetic
  ValueWithOverflow Negate() const;
  ValueWithOverflow ABS() const;

  ValueWithCarry AddUnsigned(
      const IntegerValueImpl &y, bool carryIn = false) const;
  ValueWithOverflow AddSigned(const IntegerValueImpl &y) const;
  ValueWithOverflow SubtractSigned(const IntegerValueImpl &y) const;
  ValueWithOverflow DIM(const IntegerValueImpl &y) const;
  ValueWithOverflow SIGN(const IntegerValueImpl &sign) const;

  Product MultiplySigned(const IntegerValueImpl &y) const;
  Product MultiplyUnsigned(const IntegerValueImpl &y) const;
  QuotientWithRemainder DivideSigned(const IntegerValueImpl &y) const;
  QuotientWithRemainder DivideUnsigned(const IntegerValueImpl &y) const;
  ValueWithOverflow MODULO(const IntegerValueImpl &y) const;
  PowerWithErrors Power(const IntegerValueImpl &e) const;

  // Bitwise operations
  IntegerValueImpl NOT() const;
  IntegerValueImpl IAND(const IntegerValueImpl &y) const;
  IntegerValueImpl IOR(const IntegerValueImpl &y) const;
  IntegerValueImpl IEOR(const IntegerValueImpl &y) const;
  IntegerValueImpl MERGE_BITS(
      const IntegerValueImpl &y, const IntegerValueImpl &mask) const;
  IntegerValueImpl MAX(const IntegerValueImpl &y) const {
    return CompareSigned(y) == Ordering::Less ? y : *this;
  }
  IntegerValueImpl MIN(const IntegerValueImpl &y) const {
    return CompareSigned(y) == Ordering::Less ? *this : y;
  }

  // Shift operations
  IntegerValueImpl ISHFT(int count) const {
    return count < 0 ? SHIFTR(-count) : SHIFTL(count);
  }
  IntegerValueImpl SHIFTL(int count) const;
  IntegerValueImpl SHIFTR(int count) const;
  IntegerValueImpl SHIFTA(int count) const;
  IntegerValueImpl ISHFTC(int count, int size) const;
  IntegerValueImpl ISHFTC(int count) const { return ISHFTC(count, bits()); }
  IntegerValueImpl IBITS(int pos, int size) const;
  IntegerValueImpl IBSET(int pos) const;
  IntegerValueImpl IBCLR(int pos) const;
  IntegerValueImpl DSHIFTL(const IntegerValueImpl &fill, int count) const;
  IntegerValueImpl DSHIFTR(const IntegerValueImpl &v2, int count) const;
  bool BTEST(int pos) const;
  int LEADZ() const;
  int TRAILZ() const;
  int POPCNT() const;
  bool POPPAR() const;

  static ValueWithOverflow ConvertSigned(
      int toKind, const IntegerValueImpl &from);
  static ValueWithOverflow ConvertUnsigned(
      int toKind, const IntegerValueImpl &from);

  static ValueWithOverflow Read(
      int kind, const char *&pp, int base, bool isSigned);

  // Formatting
  std::string SignedDecimal() const;
  std::string UnsignedDecimal() const;
  std::string Hexadecimal() const;

  // y converted (sign-preserving) to T, so that binary operations operate on
  // operands of equal width.  A monostate operand is treated as a zero of
  // that width.
  template <typename T> static T Coerce(const IntegerValueImpl &y) {
    if (y.IsMonostate()) {
      return T{};
    }
    return y.withWord([](const auto &yv) -> T {
      using S = std::decay_t<decltype(yv)>;
      if constexpr (std::is_same_v<S, T>) {
        return yv;
      } else {
        return T::template ConvertSigned<S>(yv).value;
      }
    });
  }

  // Same as Coerce, but zero-extending rather than sign-extending.
  template <typename T> static T CoerceUnsigned(const IntegerValueImpl &y) {
    if (y.IsMonostate()) {
      return T{};
    }
    return y.withWord([](const auto &yv) -> T {
      using S = std::decay_t<decltype(yv)>;
      if constexpr (std::is_same_v<S, T>) {
        return yv;
      } else {
        return T::template ConvertUnsigned<S>(yv).value;
      }
    });
  }

  void StoreRawBytes(void *dst, size_t size, bool *changed) const;

  // Compile-time dispatchers to current/specified kind

  template <typename F>
  auto withWordProto(F &&f) const
      -> decltype(std::declval<F>()(std::declval<I64>())) {
    return withWordProto(kind(), std::forward<F>(f));
  }

  template <typename F>
  static auto withWordProto(int kind, F &&f)
      -> decltype(std::declval<F>()(std::declval<I64>())) {
    switch (kind) {
    case 1:
      return f(I8{});
    case 2:
    case 3:
      return f(I16{});
    case 4:
      return f(I32{});
    case 8:
      return f(I64{});
    case 10:
      return f(I80{});
    case 16:
      return f(I128{});
    default:
      llvm_unreachable("unsupported integer width");
    }
  }

  template <typename F>
  auto withWord(F &&f) const
      -> decltype(std::declval<F>()(std::declval<I64>())) {
    switch (storage_.index()) {
    case 1:
      return f(std::get<I8>(storage_));
    case 2:
      return f(std::get<I16>(storage_));
    case 3:
      return f(std::get<I32>(storage_));
    case 4:
      return f(std::get<I64>(storage_));
    case 5:
      return f(std::get<I80>(storage_));
    case 6:
      return f(std::get<I128>(storage_));
    default:
      llvm_unreachable("operation on uninitialized IntegerValueImpl");
    }
  }

private:
  Storage storage_;
};

struct IntegerValueImpl::ValueWithOverflow {
  IntegerValueImpl value;
  bool overflow{false};
};

struct IntegerValueImpl::ValueWithCarry {
  IntegerValueImpl value;
  bool carry{false};
};

struct IntegerValueImpl::Product {
  IntegerValueImpl upper, lower;
  bool SignedMultiplicationOverflowed() const { return overflow; }
  bool overflow{false};
};

struct IntegerValueImpl::QuotientWithRemainder {
  IntegerValueImpl quotient, remainder;
  bool divisionByZero{false}, overflow{false};
};

struct IntegerValueImpl::PowerWithErrors {
  IntegerValueImpl power;
  bool divisionByZero{false}, overflow{false}, zeroToZero{false};
};

} // namespace Fortran::evaluate::value

namespace llvm {
/// For pretty printing in GTest
inline raw_ostream &operator<<(
    raw_ostream &os, const Fortran::evaluate::value::IntegerValueImpl &v) {
  v.print(os);
  return os;
}
} // namespace llvm

#endif // FORTRAN_EVALUATE_INTEGER_VALUE_IMPL_H_
