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

// Some environments, viz. glibc 2.17 and *BSD, allow the macro HUGE
// to leak out of <math.h>.
#undef HUGE

namespace llvm {
class raw_ostream;
}

namespace Fortran::evaluate::value {

class RealValueImpl;

// ----------------------------------------------------------------------------
// RealValue: runtime-kind IEEE floating-point value.
// ----------------------------------------------------------------------------
class RealValue {
public:
  using Word = IntegerValue;

  RealValue();
  RealValue(const RealValue &);
  RealValue(RealValue &&);
  ~RealValue();
  RealValue &operator=(const RealValue &);
  RealValue &operator=(RealValue &&);

  // Interpret w as the raw bit pattern of a value of the given runtime kind.
  RealValue(const Word &w, int kind);

  // Comparison operators
  bool operator==(const RealValue &y) const;

  // Kind-property inquiries, formerly compile-time constants derived from the
  // PREC template parameter; now selected by the runtime KIND.
  static int binaryPrecision(int kind);
  static int isImplicitMSB(int kind);
  static int DIGITS(int kind);
  static int PRECISION(int kind);
  static int RANGE(int kind);
  static int MAXEXPONENT(int kind);
  static int MINEXPONENT(int kind);

  static RealValue HUGE(int kind);
  static RealValue EPSILON(int kind);
  static RealValue TINY(int kind);
  static RealValue NotANumber(int kind);

  // Runtime kind / width accessors
  int kind() const;
  int bits() const;
  bool IsMonostate() const;
  bool IsZero() const;
  bool IsNegative() const;
  bool IsNotANumber() const;
  bool IsQuietNaN() const;
  bool IsSignalingNaN() const;
  bool IsInfinite() const;
  bool IsFinite() const;
  bool IsNormal() const;
  int Exponent() const;
  bool StoreRawBytes(void *to, std::size_t bytes) const;
  static RealValue FromRawBytes(const void *raw, std::size_t bytes);
  static RealValue Zero(int kind);

  // The raw bit pattern at the value's runtime width.
  IntegerValue RawBits() const;

  // Comparisons
  Relation Compare(const RealValue &y) const;

  // Unary operations
  RealValue ABS() const;
  RealValue Negate() const;
  RealValue SIGN(const RealValue &x) const;
  RealValue SetSign(bool toNegative) const;
  RealValue FlushSubnormalToZero() const;

  // Binary arithmetic
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
  ValueWithRealFlags<RealValue> HYPOT(const RealValue &y,
      Rounding rounding = TargetCharacteristics::defaultRounding) const;
  ValueWithRealFlags<RealValue> MOD(const RealValue &y,
      Rounding rounding = TargetCharacteristics::defaultRounding) const;
  ValueWithRealFlags<RealValue> MODULO(const RealValue &y,
      Rounding rounding = TargetCharacteristics::defaultRounding) const;
  ValueWithRealFlags<RealValue> DIM(const RealValue &y,
      Rounding rounding = TargetCharacteristics::defaultRounding) const;

  // Convert this value to a different real kind (single rounding).
  RealValue ConvertToKind(int kind) const { return Convert(*this, kind).value; }

  RealValue FRACTION() const;
  RealValue RRSPACING() const;
  RealValue SPACING() const;
  RealValue SET_EXPONENT(std::int64_t e) const;

  ValueWithRealFlags<RealValue> NEAREST(bool upward) const;
  ValueWithRealFlags<RealValue> ToWholeNumber(
      common::RoundingMode mode = common::RoundingMode::ToZero) const;
  // Convert this real to an integer of the given bit width.
  ValueWithRealFlags<IntegerValue> ToInteger(
      common::RoundingMode mode = common::RoundingMode::ToZero,
      int toBits = 0) const;

  ValueWithRealFlags<RealValue> SCALE(const IntegerValue &by,
      Rounding rounding = TargetCharacteristics::defaultRounding) const;

  ValueWithRealFlags<RealValue> KahanSummation(const RealValue &y,
      RealValue &correction,
      Rounding rounding = TargetCharacteristics::defaultRounding) const;

  IntegerValue EXPONENT() const;

  // Conversion from an integer facade (REAL()).
  static ValueWithRealFlags<RealValue> FromInteger(const IntegerValue &n,
      int kind, bool isUnsigned = false,
      Rounding rounding = TargetCharacteristics::defaultRounding);

  // Conversion between real kinds.
  static ValueWithRealFlags<RealValue> Convert(const RealValue &from, int kind,
      Rounding rounding = TargetCharacteristics::defaultRounding);

  static ValueWithRealFlags<RealValue> Read(const char *&pp, int kind,
      Rounding rounding = TargetCharacteristics::defaultRounding);

  std::string DumpHexadecimal() const;
  llvm::raw_ostream &AsFortran(
      llvm::raw_ostream &o, int kind, bool minimal = false) const;

private:
  alignas(detail::kRealObjectAlign) char opaque_[detail::kRealObjectSize];

  RealValueImpl &impl() { return *reinterpret_cast<RealValueImpl *>(this); }
  const RealValueImpl &impl() const {
    return *reinterpret_cast<const RealValueImpl *>(this);
  }

  static RealValue FromImpl(const RealValueImpl &x);
  static RealValue FromImpl(RealValueImpl &&x);
  static ValueWithRealFlags<RealValue> FromImpl(
      const ValueWithRealFlags<RealValueImpl> &x);
};

static_assert(sizeof(RealValue) == detail::kRealObjectSize);

} // namespace Fortran::evaluate::value
#endif // FORTRAN_EVALUATE_REAL_VALUE_H_
