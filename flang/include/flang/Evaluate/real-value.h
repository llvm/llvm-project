//===-- include/flang/Evaluate/real-value.h ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_EVALUATE_REAL_VALUE_H_
#define FORTRAN_EVALUATE_REAL_VALUE_H_

#include "flang/Evaluate/integer-value.h"
#include "flang/Evaluate/object-sizes.h"
#include "flang/Evaluate/target.h"
#include "llvm/Support/Compiler.h"

// Some environments, viz. glibc 2.17 and *BSD, allow the macro HUGE
// to leak out of <math.h>.
#undef HUGE

namespace Fortran::evaluate::value {
class RealValueImpl;

/// A floating-point value with dynamic precision.
///
/// The precision is dynamic, but only a predefined set of Fortran kinds are
/// allowed. It is also kind-aware, i.e. knows which REAL kind it currently
/// represents.
///
/// The implementation is hidden from this header using a pImpl-like idiom.
class RealValue {
public:
  using Word = IntegerValue;

  RealValue();
  ~RealValue();
  RealValue(const RealValue &);
  RealValue(RealValue &&);
  RealValue &operator=(const RealValue &);
  RealValue &operator=(RealValue &&);

  RealValue(int kind, const RealValue &v) : RealValue(v) {
    CHECK(kind == v.kind());
  }
  RealValue(int kind, RealValue &&v) : RealValue(std::move(v)) {
    CHECK(kind == v.kind());
  }

  /// Interpret w as the raw bit pattern for the given runtime kind.
  RealValue(int kind, const Word &w);

  /// Creates a floating-point value of a given kind from a host double,
  /// rounded to the target kind's precision (per the default rounding mode).
  /// Portable: does not assume that the host "double" shares any bit layout
  /// with the target kind, only that <cmath>'s frexp()/ldexp() are available.
  RealValue(int kind, double x);

  /// Creates a floating-point with value +0.0 of a given kind. In contrast, the
  /// default ctor creates a "monostate" that represents +0.0 of unknown kind.
  static RealValue Zero(int kind);

  /// Creates a floating-point with value -0.0 of a given kind.
  static RealValue NegativeZero(int kind);

  static RealValue Infinity(int kind, bool negative = false);

  void print(llvm::raw_ostream &os) const;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump() const;
#endif

  /// Whether this object represents a default-initialized value (zero) or
  /// unknown value.
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

  bool operator==(const RealValue &y) const;
  bool operator!=(const RealValue &y) const { return !operator==(y); }

  bool IsNegative() const;

  bool IsNotANumber() const;

  bool IsSignalingNaN() const;

  bool IsInfinite() const;

  bool IsFinite() const;

  bool IsZero() const;

  bool IsNormal() const;

  RealValue ABS() const;

  RealValue SetSign(bool toNegative) const;

  RealValue SIGN(const RealValue &x) const;

  RealValue Negate() const;

  Relation Compare(const RealValue &y) const;

  ValueWithRealFlags<RealValue> Add(const RealValue &y,
      Rounding rounding = TargetCharacteristics::defaultRounding) const;

  ValueWithRealFlags<RealValue> Subtract(const RealValue &y,
      Rounding rounding = TargetCharacteristics::defaultRounding) const;

  ValueWithRealFlags<RealValue> Multiply(const RealValue &y,
      Rounding rounding = TargetCharacteristics::defaultRounding) const;

  ValueWithRealFlags<RealValue> Divide(const RealValue &y,
      Rounding rounding = TargetCharacteristics::defaultRounding) const;

  ValueWithRealFlags<RealValue> SQRT(
      Rounding rounding = TargetCharacteristics::defaultRounding) const;

  ///  NEAREST(), IEEE_NEXT_AFTER(), IEEE_NEXT_UP(), and IEEE_NEXT_DOWN()
  ValueWithRealFlags<RealValue> NEAREST(bool upward) const;

  /// HYPOT(x,y)=SQRT(x**2 + y**2) computed so as to avoid spurious
  /// intermediate overflows.
  ValueWithRealFlags<RealValue> HYPOT(const RealValue &y,
      Rounding rounding = TargetCharacteristics::defaultRounding) const;

  /// DIM(X,Y) = MAX(X-Y, 0)
  ValueWithRealFlags<RealValue> DIM(const RealValue &y,
      Rounding rounding = TargetCharacteristics::defaultRounding) const;

  /// MOD(x,y) = x - AINT(x/y)*y (in the standard)
  ValueWithRealFlags<RealValue> MOD(const RealValue &y,
      Rounding rounding = TargetCharacteristics::defaultRounding) const;

  /// MODULO(x,y) = x - FLOOR(x/y)*y (in the standard)
  ValueWithRealFlags<RealValue> MODULO(const RealValue &y,
      Rounding rounding = TargetCharacteristics::defaultRounding) const;

  ValueWithRealFlags<RealValue> KahanSummation(const RealValue &y,
      RealValue &correction,
      Rounding rounding = TargetCharacteristics::defaultRounding) const;

  IntegerValue EXPONENT() const;

  static RealValue EPSILON(int kind);

  static RealValue HUGE(int kind);

  static RealValue TINY(int kind);

  static int DIGITS(int kind);

  static int PRECISION(int kind);

  static int RANGE(int kind);

  static int MAXEXPONENT(int kind);

  static int MINEXPONENT(int kind);

  RealValue RRSPACING() const;

  RealValue SPACING() const;

  RealValue SET_EXPONENT(std::int64_t e) const;

  RealValue FRACTION() const;

  /// SCALE(); also known as IEEE_SCALB and (in IEEE-754 '08) ScaleB.
  ValueWithRealFlags<RealValue> SCALE(const IntegerValue &by,
      Rounding rounding = TargetCharacteristics::defaultRounding) const;

  RealValue FlushSubnormalToZero() const;

  // TODO: Configurable NotANumber representations
  static RealValue NotANumber(int kind);

  static ValueWithRealFlags<RealValue> FromInteger(int kind,
      const IntegerValue &n, bool isUnsigned = false,
      Rounding rounding = TargetCharacteristics::defaultRounding);

  /// Conversion to integer in the same real format (AINT(), ANINT())
  ValueWithRealFlags<RealValue> ToWholeNumber(
      common::RoundingMode mode = common::RoundingMode::ToZero) const;

  /// Conversion to an integer (INT(), NINT(), FLOOR(), CEILING())
  ValueWithRealFlags<IntegerValue> ToInteger(
      common::RoundingMode mode = common::RoundingMode::ToZero,
      int toBits = 0) const;

  static ValueWithRealFlags<RealValue> Convert(int kind, const RealValue &from,
      Rounding rounding = TargetCharacteristics::defaultRounding);

  Word RawBits() const;

  /// Extracts "raw" biased exponent field.
  int Exponent() const;

  static ValueWithRealFlags<RealValue> Read(int kind, const char *&pp,
      Rounding rounding = TargetCharacteristics::defaultRounding);

  std::string DumpHexadecimal() const;

  /// Emits a character representation for an equivalent Fortran constant
  /// or parenthesized constant expression that produces this value.
  llvm::raw_ostream &AsFortran(
      llvm::raw_ostream &o, int kind, bool minimal = false) const;

  static RealValue FromRawBytes(
      int kind, const void *raw, std::size_t expectedSize);

  void StoreRawBytes(void *dst, size_t size, bool *changed = nullptr) const;

private:
  static RealValue FromImpl(const RealValueImpl &x);
  static RealValue FromImpl(RealValueImpl &&x);
  static ValueWithRealFlags<RealValue> FromImpl(
      const ValueWithRealFlags<RealValueImpl> &x);
  static ValueWithRealFlags<RealValue> FromImpl(
      ValueWithRealFlags<RealValueImpl> &&x);

  RealValueImpl &impl() { return *reinterpret_cast<RealValueImpl *>(this); }
  const RealValueImpl &impl() const {
    return *reinterpret_cast<const RealValueImpl *>(this);
  }

  [[maybe_unused]] alignas(
      detail::kRealObjectAlign) char opaque_[detail::kRealObjectSize];
};

} // namespace Fortran::evaluate::value

namespace llvm {
/// For pretty printing in GTest
inline raw_ostream &operator<<(
    raw_ostream &os, const Fortran::evaluate::value::RealValue &v) {
  v.print(os);
  return os;
}
} // namespace llvm

#endif // FORTRAN_EVALUATE_REAL_VALUE_H_
