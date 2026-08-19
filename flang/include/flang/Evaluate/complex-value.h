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
#include "llvm/Support/Compiler.h"

namespace llvm {
class raw_ostream;
}

namespace Fortran::evaluate::value {
using common::KindsEnum;

/// A complex floating-point value with dynamic precision.
///
/// The precision is dynamic, but only a predefined set of Fortran kinds are
/// allowed. It is also kind-aware, i.e. knows which COMPLEX kind it currently
/// represents.
///
/// The implementation is a pair of RealValue objects.
class ComplexValue {
public:
  ComplexValue() = default;
  ComplexValue(const ComplexValue &) = default;
  ComplexValue(ComplexValue &&) = default;
  ComplexValue &operator=(const ComplexValue &) = default;
  ComplexValue &operator=(ComplexValue &&) = default;

  ComplexValue(const RealValue &r, const RealValue &i)
      : re_{r},
        im_{r.IsMonostate() ? i : RealValue::Convert(r.kind(), i).value} {}

  explicit ComplexValue(const RealValue &r)
      : ComplexValue{r, RealValue::Zero(r.kind())} {}

  ComplexValue(KindsEnum kind, const RealValue &r) : ComplexValue{r} {
    CHECK(kind == r.kind());
  }

  ComplexValue(KindsEnum kind, const ComplexValue &v) : ComplexValue{v} {
    CHECK(kind == v.kind());
  }

  ComplexValue(KindsEnum kind, ComplexValue &&v) : ComplexValue{std::move(v)} {
    CHECK(kind == v.kind());
  }

  /// Creates a complex value (+0.0 + +0.0i) of a given kind. This is
  /// different from the default-ctor which creates a "monostate" that
  /// represents zero of unknown kind.
  static ComplexValue Zero(KindsEnum kind) {
    RealValue zero{RealValue::Zero(kind)};
    return ComplexValue{zero, zero};
  }

  void print(llvm::raw_ostream &os) const;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump() const;
#endif

  /// Whether this object represents a default-initialized value (zero) of
  /// not-yet-known kind.
  bool IsMonostate() const {
    CHECK(re_.IsMonostate() == im_.IsMonostate());
    return re_.IsMonostate();
  }

  /// The kind of the value currently stored.
  KindsEnum kind() const {
    CHECK(re_.kind() == im_.kind());
    return re_.kind();
  }

  /// Number of bytes accessed by FromRawBytes/StoreRawBytes
  std::size_t bytesStored() const {
    return re_.bytesStored() + im_.bytesStored();
  }
  static constexpr std::size_t bytesStored(KindsEnum kind) {
    return 2 * RealValue::bytesStored(kind);
  }

  RealValue REAL() const { return re_; }

  RealValue AIMAG() const { return im_; }

  ComplexValue CONJG() const { return ComplexValue{re_, im_.Negate()}; }

  ComplexValue Negate() const {
    return ComplexValue{re_.Negate(), im_.Negate()};
  }

  bool Equals(const ComplexValue &y) const {
    return re_.Compare(y.re_) == Relation::Equal &&
        im_.Compare(y.im_) == Relation::Equal;
  }

  bool operator==(const ComplexValue &y) const {
    return re_ == y.re_ && im_ == y.im_;
  }

  bool operator!=(const ComplexValue &y) const { return !(*this == y); }

  bool IsZero() const { return re_.IsZero() && im_.IsZero(); }

  bool IsInfinite() const { return re_.IsInfinite() || im_.IsInfinite(); }

  bool IsNotANumber() const { return re_.IsNotANumber() || im_.IsNotANumber(); }

  bool IsSignalingNaN() const {
    return re_.IsSignalingNaN() || im_.IsSignalingNaN();
  }

  static ValueWithRealFlags<ComplexValue> FromInteger(KindsEnum kind,
      const IntegerValue &n, bool isUnsigned = false,
      Rounding rounding = TargetCharacteristics::defaultRounding);

  ValueWithRealFlags<ComplexValue> Add(const ComplexValue &,
      Rounding rounding = TargetCharacteristics::defaultRounding) const;

  ValueWithRealFlags<ComplexValue> Subtract(const ComplexValue &,
      Rounding rounding = TargetCharacteristics::defaultRounding) const;
  ValueWithRealFlags<ComplexValue> Multiply(const ComplexValue &,
      Rounding rounding = TargetCharacteristics::defaultRounding) const;
  ValueWithRealFlags<ComplexValue> Divide(const ComplexValue &,
      Rounding rounding = TargetCharacteristics::defaultRounding) const;

  ValueWithRealFlags<ComplexValue> KahanSummation(const ComplexValue &y,
      ComplexValue &correction,
      Rounding rounding = TargetCharacteristics::defaultRounding) const;

  /// ABS/CABS = HYPOT(re_, imag_) = SQRT(re_**2 + im_**2)
  ValueWithRealFlags<RealValue> ABS(
      Rounding rounding = TargetCharacteristics::defaultRounding) const {
    return re_.HYPOT(im_, rounding);
  }

  ComplexValue FlushSubnormalToZero() const {
    return ComplexValue{re_.FlushSubnormalToZero(), im_.FlushSubnormalToZero()};
  }

  static ComplexValue NotANumber(KindsEnum kind) {
    return {RealValue::NotANumber(kind), RealValue::NotANumber(kind)};
  }

  std::string DumpHexadecimal() const;

  llvm::raw_ostream &AsFortran(llvm::raw_ostream &, int kind) const;

  void StoreRawBytes(void *dst, size_t size, bool *changed = nullptr) const;

  static ComplexValue FromRawBytes(
      KindsEnum kind, const void *raw, std::size_t expectedSize);

  // TODO: unit testing

private:
  RealValue re_, im_;
};

} // namespace Fortran::evaluate::value

namespace llvm {
/// For pretty printing in GTest
inline raw_ostream &operator<<(
    raw_ostream &os, const Fortran::evaluate::value::ComplexValue &v) {
  v.print(os);
  return os;
}
} // namespace llvm

#endif // FORTRAN_EVALUATE_COMPLEX_VALUE_H_
