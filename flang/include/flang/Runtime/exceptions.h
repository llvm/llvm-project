//===-- include/flang/Runtime/exceptions.h ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Map Fortran ieee_arithmetic module exceptions to fenv.h exceptions.

#ifndef FORTRAN_RUNTIME_EXCEPTIONS_H_
#define FORTRAN_RUNTIME_EXCEPTIONS_H_

#include "flang/Runtime/entry-names.h"
#include <cinttypes>

namespace Fortran::runtime {

class Descriptor;

extern "C" {

// Map a set of IEEE_FLAG_TYPE exception values to a libm fenv.h excepts value.
// This mapping is done at runtime to support cross compilation.
std::uint32_t RTNAME(MapException)(std::uint32_t excepts);

// Check if the processor has the ability to control whether to halt
// or continue exeuction when a given exception is raised.
bool RTNAME(SupportHalting)(uint32_t except);

// Get and set the ieee underflow mode if supported; otherwise nops.
bool RTNAME(GetUnderflowMode)(void);
void RTNAME(SetUnderflowMode)(bool flag);

} // extern "C"
} // namespace Fortran::runtime
#endif // FORTRAN_RUNTIME_EXCEPTIONS_H_
