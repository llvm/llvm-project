//===-- include/flang/Evaluate/complex.h ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_EVALUATE_COMPLEX_H_
#define FORTRAN_EVALUATE_COMPLEX_H_

#include "formatting.h"
#include "real.h"
#include <string>

namespace llvm {
class raw_ostream;
}

namespace Fortran::evaluate::value {

template <typename REAL_TYPE> class Complex {
public:
  using Part = REAL_TYPE;
  static constexpr int bits{2 * Part::bits};

  constexpr Complex() {} // (+0.0, +0.0)
  constexpr Complex(const Complex &) = default;
  constexpr Complex(const Part &r, const Part &i) : re_{r}, im_{i} {}
  explicit constexpr Complex(const Part &r) : re_{r} {}
  constexpr Complex &operator=(const Complex &) = default;
  constexpr Complex &operator=(Complex &&) = default;

  constexpr bool operator==(const Complex &that) const {
    return re_ == that.re_ && im_ == that.im_;
  }

  constexpr const Part &REAL() const { return re_; }
  constexpr const Part &AIMAG() const { return im_; }
  constexpr Complex CONJG() const { return {re_, im_.Negate()}; }
  constexpr Complex Negate() const { return {re_.Negate(), im_.Negate()}; }

  constexpr bool Equals(const Complex &that) const {
    return re_.Compare(that.re_) == Relation::Equal &&
        im_.Compare(that.im_) == Relation::Equal;
  }

  constexpr bool IsZero() const { return re_.IsZero() || im_.IsZero(); }

  constexpr bool IsInfinite() const {
    return re_.IsInfinite() || im_.IsInfinite();
  }

  constexpr bool IsNotANumber() const {
    return re_.IsNotANumber() || im_.IsNotANumber();
  }

  constexpr bool IsSignalingNaN() const {
    return re_.IsSignalingNaN() || im_.IsSignalingNaN();
  }

  template <typename INT>
  static ValueWithRealFlags<Complex> FromInteger(const INT &n,
      bool isUnsigned = false,
      Rounding rounding = TargetCharacteristics::defaultRounding) {
    ValueWithRealFlags<Complex> result;
    result.value.re_ = Part::FromInteger(n, isUnsigned, rounding)
                           .AccumulateFlags(result.flags);
    return result;
  }

  ValueWithRealFlags<Complex> Add(const Complex &,
      Rounding rounding = TargetCharacteristics::defaultRounding) const;
  ValueWithRealFlags<Complex> Subtract(const Complex &,
      Rounding rounding = TargetCharacteristics::defaultRounding) const;
  ValueWithRealFlags<Complex> Multiply(const Complex &,
      Rounding rounding = TargetCharacteristics::defaultRounding) const;
  ValueWithRealFlags<Complex> Divide(const Complex &,
      Rounding rounding = TargetCharacteristics::defaultRounding) const;

  // ABS/CABS = HYPOT(re_, imag_) = SQRT(re_**2 + im_**2)
  ValueWithRealFlags<Part> ABS(
      Rounding rounding = TargetCharacteristics::defaultRounding) const {
    return re_.HYPOT(im_, rounding);
  }

  constexpr Complex FlushSubnormalToZero() const {
    return {re_.FlushSubnormalToZero(), im_.FlushSubnormalToZero()};
  }

  static constexpr Complex NotANumber() {
    return {Part::NotANumber(), Part::NotANumber()};
  }

  std::string DumpHexadecimal() const;
  llvm::raw_ostream &AsFortran(llvm::raw_ostream &, int kind) const;

  // TODO: unit testing

private:
  Part re_, im_;
};

extern template class Complex<Real<Integer<16>, 11>>;
extern template class Complex<Real<Integer<16>, 8>>;
extern template class Complex<Real<Integer<32>, 24>>;
extern template class Complex<Real<Integer<64>, 53>>;
extern template class Complex<Real<X87IntegerContainer, 64>>;
extern template class Complex<Real<Integer<128>, 113>>;
} // namespace Fortran::evaluate::value
#endif // FORTRAN_EVALUATE_COMPLEX_H_
