//===-- lib/Evaluate/int-power.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_EVALUATE_INT_POWER_H_
#define FORTRAN_EVALUATE_INT_POWER_H_

// Computes an integer power of a real or complex value.

#include "flang/Evaluate/real-value.h"
#include "flang/Evaluate/real.h"
#include "flang/Evaluate/target.h"

namespace Fortran::evaluate {

template <typename REAL>
ValueWithRealFlags<REAL> TimesIntPowerOf(const REAL &factor, const REAL &base,
    const value::IntegerValue &power, bool forceMultiply = false,
    Rounding rounding = TargetCharacteristics::defaultRounding) {
  using INT = value::IntegerValue;
  const int realKind{base.kind()};
  CHECK(factor.kind() == base.kind());
  ValueWithRealFlags<REAL> result{factor};
  if (base.IsNotANumber()) {
    result.value = REAL::NotANumber(realKind);
    result.flags.set(RealFlag::InvalidArgument);
  } else if (power.IsZero()) {
    if (base.IsZero() || base.IsInfinite()) {
      result.flags.set(RealFlag::InvalidArgument);
    }
  } else {
    bool negativePower{!forceMultiply && power.IsNegative()};
    INT absPower{power.ABS().value};
    REAL squares{base};
    int nbits{absPower.bits() - absPower.LEADZ()};
    for (int j{0}; j < nbits; ++j) {
      if (j > 0) { // avoid spurious overflow on last iteration
        squares =
            squares.Multiply(squares, rounding).AccumulateFlags(result.flags);
      }
      if (absPower.BTEST(j)) {
        if (negativePower) {
          result.value = result.value.Divide(squares, rounding)
                             .AccumulateFlags(result.flags);
        } else {
          result.value = result.value.Multiply(squares, rounding)
                             .AccumulateFlags(result.flags);
        }
      }
    }
  }
  return result;
}

inline ValueWithRealFlags<value::RealValue> IntPower(
    const value::RealValue &base, const value::IntegerValue &power,
    Rounding rounding = TargetCharacteristics::defaultRounding) {
  using REAL = value::RealValue;
  using INT = value::IntegerValue;
  const int realKind{base.kind()};
  const int intKind{power.kind()};
  REAL one{REAL::FromInteger(realKind, INT{intKind, 1}).value};
  ValueWithRealFlags<REAL> result{
      TimesIntPowerOf(one, base, power, /*forceMultiply=*/false, rounding)};
  if (result.flags.test(RealFlag::Overflow) && power.IsNegative() &&
      !base.IsZero() && base.ABS().Compare(one) == Relation::Greater) {
    ValueWithRealFlags<REAL> retry;
    REAL recip{one.Divide(base, rounding).AccumulateFlags(retry.flags)};
    retry.value =
        TimesIntPowerOf(one, recip, power, /*forceMultiply=*/true, rounding)
            .AccumulateFlags(retry.flags);
    return retry;
  }
  return result;
}

inline ValueWithRealFlags<value::ComplexValue> IntPower(
    const value::ComplexValue &base, const value::IntegerValue &power,
    Rounding rounding = TargetCharacteristics::defaultRounding) {
  using CMPLX = value::ComplexValue;
  using INT = value::IntegerValue;
  const int cmplxKind{base.kind()};
  const int intKind{power.kind()};
  CMPLX one{CMPLX::FromInteger(cmplxKind, INT{intKind, 1}).value};
  return TimesIntPowerOf(one, base, power, /*forceMultiply=*/false, rounding);
}

} // namespace Fortran::evaluate
#endif // FORTRAN_EVALUATE_INT_POWER_H_
