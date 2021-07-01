//===-- runtime/time-intrinsic.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Defines the API between compiled code and the implementations of time-related
// intrinsic subroutines in the runtime library.

#ifndef FORTRAN_RUNTIME_TIME_INTRINSIC_H_
#define FORTRAN_RUNTIME_TIME_INTRINSIC_H_

#include "cpp-type.h"
#include "entry-names.h"

namespace Fortran::runtime {
extern "C" {

// Lowering may need to cast this result to match the precision of the default
// real kind.
double RTNAME(CpuTime)();

// Interface for the SYSTEM_CLOCK intrinsic. We break it up into 3 distinct
// function calls, one for each of SYSTEM_CLOCK's optional output arguments.
// Lowering will have to cast the results to whatever type it prefers.
CppTypeFor<TypeCategory::Integer, 8> RTNAME(SystemClockCount)();
CppTypeFor<TypeCategory::Integer, 8> RTNAME(SystemClockCountRate)();
CppTypeFor<TypeCategory::Integer, 8> RTNAME(SystemClockCountMax)();
} // extern "C"
} // namespace Fortran::runtime
#endif // FORTRAN_RUNTIME_TIME_INTRINSIC_H_
