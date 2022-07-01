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

#include "flang/Evaluate/target.h"

namespace Fortran::evaluate {

template <typename REAL, typename INT>
ValueWithRealFlags<REAL> TimesIntPowerOf(const REAL &factor, const REAL &base,
    const INT &power,
    Rounding rounding = TargetCharacteristics::defaultRounding) {
  ValueWithRealFlags<REAL> result{factor};
  if (base.IsNotANumber()) {
    result.value = REAL::NotANumber();
    result.flags.set(RealFlag::InvalidArgument);
  } else if (power.IsZero()) {
    if (base.IsZero() || base.IsInfinite()) {
      result.flags.set(RealFlag::InvalidArgument);
    }
  } else {
    bool negativePower{power.IsNegative()};
    INT absPower{power.ABS().value};
    REAL squares{base};
    int nbits{INT::bits - absPower.LEADZ()};
    for (int j{0}; j < nbits; ++j) {
      if (absPower.BTEST(j)) {
        if (negativePower) {
          result.value = result.value.Divide(squares, rounding)
                             .AccumulateFlags(result.flags);
        } else {
          result.value = result.value.Multiply(squares, rounding)
                             .AccumulateFlags(result.flags);
        }
      }
      squares =
          squares.Multiply(squares, rounding).AccumulateFlags(result.flags);
    }
  }
  return result;
}

template <typename REAL, typename INT>
ValueWithRealFlags<REAL> IntPower(const REAL &base, const INT &power,
    Rounding rounding = TargetCharacteristics::defaultRounding) {
  REAL one{REAL::FromInteger(INT{1}).value};
  return TimesIntPowerOf(one, base, power, rounding);
}
} // namespace Fortran::evaluate
#endif // FORTRAN_EVALUATE_INT_POWER_H_
