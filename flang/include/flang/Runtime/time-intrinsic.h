//===-- include/flang/Runtime/time-intrinsic.h ------------------*- C++ -*-===//
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

#include "flang/Runtime/entry-names.h"
#include <cinttypes>
#include <cstddef>

namespace Fortran::runtime {

class Descriptor;

extern "C" {

// Lowering may need to cast this result to match the precision of the default
// real kind.
double RTNAME(CpuTime)();

// Interface for the SYSTEM_CLOCK intrinsic. We break it up into 3 distinct
// function calls, one for each of SYSTEM_CLOCK's optional output arguments.
// Lowering converts the results to the types of the actual arguments,
// including the case of a real argument for COUNT_RATE=..
// The kind argument to SystemClockCount and SystemClockCountMax is the
// kind of the integer actual arguments, which are required to be the same
// when both appear.
std::int64_t RTNAME(SystemClockCount)(int kind = 8);
std::int64_t RTNAME(SystemClockCountRate)(int kind = 8);
std::int64_t RTNAME(SystemClockCountMax)(int kind = 8);

// Interface for DATE_AND_TIME intrinsic.
void RTNAME(DateAndTime)(char *date, std::size_t dateChars, char *time,
    std::size_t timeChars, char *zone, std::size_t zoneChars,
    const char *source = nullptr, int line = 0,
    const Descriptor *values = nullptr);

void RTNAME(Etime)(const Descriptor *values, const Descriptor *time,
    const char *sourceFile, int line);

} // extern "C"
} // namespace Fortran::runtime
#endif // FORTRAN_RUNTIME_TIME_INTRINSIC_H_
