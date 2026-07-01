//===-- include/flang/Evaluate/complex-value.h ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_EVALUATE_COMPLEX_VALUE_H_
#define FORTRAN_EVALUATE_COMPLEX_VALUE_H_

#include "real-value.h"

namespace llvm {
class raw_ostream;
}

namespace Fortran::evaluate::value {

// ----------------------------------------------------------------------------
// ComplexValue: pair of RealValue components.
// ----------------------------------------------------------------------------
class ComplexValue {
public:
  ComplexValue() = default;
  ComplexValue(const RealValue &r, const RealValue &i)
      : re_{r}, im_{r.IsMonostate() ? i : i.ConvertToKind(r.kind())} {}
  explicit ComplexValue(const RealValue &r)
      : ComplexValue{r, RealValue::Zero(r.kind())} {}

  // Comparison operators
  bool operator==(const ComplexValue &y) const;
  bool operator!=(const ComplexValue &y) const { return !(*this == y); }

  static ComplexValue NotANumber(int kind) {
    return {RealValue::NotANumber(kind), RealValue::NotANumber(kind)};
  }

  // Runtime kind / width accessors
  int kind() const;
  int bits() const;
  // A COMPLEX is the former monostate iff its real part is.
  bool IsMonostate() const { return re_.IsMonostate(); }
  bool IsZero() const;
  bool IsInfinite() const;
  bool IsNotANumber() const;
  bool IsSignalingNaN() const;
  bool Equals(const ComplexValue &y) const;
  bool StoreRawBytes(void *to, std::size_t strideBytes) const;
  static ComplexValue FromRawBytes(const void *raw, std::size_t bytes);
  static ComplexValue Zero(int kind);

  RealValue REAL() const;
  RealValue AIMAG() const;

  // Unary operations
  ComplexValue CONJG() const;
  ComplexValue Negate() const;
  ComplexValue FlushSubnormalToZero() const;

  // Binary arithmetic
  ValueWithRealFlags<ComplexValue> Add(const ComplexValue &,
      Rounding rounding = TargetCharacteristics::defaultRounding) const;
  ValueWithRealFlags<ComplexValue> Subtract(const ComplexValue &,
      Rounding rounding = TargetCharacteristics::defaultRounding) const;
  ValueWithRealFlags<ComplexValue> Multiply(const ComplexValue &,
      Rounding rounding = TargetCharacteristics::defaultRounding) const;
  ValueWithRealFlags<ComplexValue> Divide(const ComplexValue &,
      Rounding rounding = TargetCharacteristics::defaultRounding) const;
  ValueWithRealFlags<RealValue> ABS(
      Rounding rounding = TargetCharacteristics::defaultRounding) const;

  ComplexValue ConvertToKind(int kind) const;

  static ValueWithRealFlags<ComplexValue> FromInteger(const IntegerValue &n,
      int kind, bool isUnsigned = false,
      Rounding rounding = TargetCharacteristics::defaultRounding);

  ValueWithRealFlags<ComplexValue> KahanSummation(const ComplexValue &y,
      ComplexValue &correction,
      Rounding rounding = TargetCharacteristics::defaultRounding) const;

  std::string DumpHexadecimal() const;
  llvm::raw_ostream &AsFortran(llvm::raw_ostream &, int kind) const;

private:
  RealValue re_;
  RealValue im_;
};

} // namespace Fortran::evaluate::value
#endif // FORTRAN_EVALUATE_COMPLEX_VALUE_H_
