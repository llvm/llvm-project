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

#include "flang/Evaluate/complex-value.h"
#include "flang/Evaluate/real-value.h"
#include "flang/Evaluate/target.h"

namespace Fortran::evaluate {

ValueWithRealFlags<value::RealValue> IntPower(const value::RealValue &base,
    const value::IntegerValue &power,
    Rounding rounding = TargetCharacteristics::defaultRounding);

ValueWithRealFlags<value::ComplexValue> IntPower(
    const value::ComplexValue &base, const value::IntegerValue &power,
    Rounding rounding = TargetCharacteristics::defaultRounding);

} // namespace Fortran::evaluate
#endif // FORTRAN_EVALUATE_INT_POWER_H_
