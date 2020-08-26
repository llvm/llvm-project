//===-- runtime/clock.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Defines API between compiled code and the time measurement
// support functions in the runtime library.

#ifndef FORTRAN_RUNTIME_CLOCK_H_
#define FORTRAN_RUNTIME_CLOCK_H_
#include "entry-names.h"
#include <cstddef>

namespace Fortran::runtime {

class Descriptor;

extern "C" {

/// Implement runtime for DATE_AND_TIME intrinsic.
/// TODO:
/// - Add VALUES argument (through descriptor).
/// - Windows implementation (currently does nothing)
void RTNAME(DateAndTime)(char *date, char *time, char *zone,
    std::size_t dateChars, std::size_t timeChars, std::size_t zoneChars);
}

// TODO: CPU_TIME, SYSTEM_CLOCK
} // namespace Fortran::runtime
#endif // FORTRAN_RUNTIME_CLOCK_H_
