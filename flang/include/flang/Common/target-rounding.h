//===-- include/flang/Common/target-rounding.h ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_COMMON_TARGET_ROUNDING_H_
#define FORTRAN_COMMON_TARGET_ROUNDING_H_

#include "flang/Common/Fortran-consts.h"
#include "flang/Common/enum-set.h"

namespace Fortran::common {

// Floating-point rounding control
struct Rounding {
  common::RoundingMode mode{common::RoundingMode::TiesToEven};
  // When set, emulate status flag behavior peculiar to x86
  // (viz., fail to set the Underflow flag when an inexact product of a
  // multiplication is rounded up to a normal number from a subnormal
  // in some rounding modes)
#if __x86_64__ || __riscv || __loongarch__
  bool x86CompatibleBehavior{true};
#else
  bool x86CompatibleBehavior{false};
#endif
};

// These are ordered like the bits in a common fenv.h header file.
ENUM_CLASS(RealFlag, InvalidArgument, Denorm, DivideByZero, Overflow, Underflow,
    Inexact)
using RealFlags = common::EnumSet<RealFlag, RealFlag_enumSize>;

} // namespace Fortran::common
#endif /* FORTRAN_COMMON_TARGET_ROUNDING_H_ */
