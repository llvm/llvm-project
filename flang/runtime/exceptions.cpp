//===-- runtime/exceptions.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Map Fortran ieee_arithmetic module exceptions to fenv.h exceptions.

#include "flang/Runtime/exceptions.h"
#include "terminator.h"
#include "flang/Runtime/magic-numbers.h"
#include <cfenv>

#ifndef __FE_DENORM
#define __FE_DENORM 0 // denorm is nonstandard
#endif

namespace Fortran::runtime {

extern "C" {

std::int32_t RTNAME(MapException)(int32_t except) {
  Terminator terminator{__FILE__, __LINE__};

  static constexpr int32_t mask{_FORTRAN_RUNTIME_IEEE_INVALID |
      _FORTRAN_RUNTIME_IEEE_DENORM | _FORTRAN_RUNTIME_IEEE_DIVIDE_BY_ZERO |
      _FORTRAN_RUNTIME_IEEE_OVERFLOW | _FORTRAN_RUNTIME_IEEE_UNDERFLOW |
      _FORTRAN_RUNTIME_IEEE_INEXACT};
  if (except == 0 || except != (except & mask)) {
    terminator.Crash("Invalid exception value: %d", except);
  }

  // Fortran and fenv.h values are identical; return the value.
  if constexpr (_FORTRAN_RUNTIME_IEEE_INVALID == FE_INVALID &&
      _FORTRAN_RUNTIME_IEEE_DENORM == __FE_DENORM &&
      _FORTRAN_RUNTIME_IEEE_DIVIDE_BY_ZERO == FE_DIVBYZERO &&
      _FORTRAN_RUNTIME_IEEE_OVERFLOW == FE_OVERFLOW &&
      _FORTRAN_RUNTIME_IEEE_UNDERFLOW == FE_UNDERFLOW &&
      _FORTRAN_RUNTIME_IEEE_INEXACT == FE_INEXACT) {
    return except;
  }

  // fenv.h calls that take exception arguments are able to process multiple
  // exceptions in one call, such as FE_OVERFLOW | FE_DIVBYZERO | FE_INVALID.
  // And intrinsic module procedures that manage exceptions are elemental
  // procedures that may specify multiple exceptions, such as ieee_all.
  // However, general elemental call processing places single scalar arguments
  // in a loop. As a consequence, argument 'except' here will be a power of
  // two, corresponding to a single exception. If code generation were
  // modified to bypass normal elemental call processing for calls with
  // ieee_usual, ieee_all, or user-specified array arguments, this switch
  // could be extended to support that.

  // Fortran and fenv.h values differ.
  switch (except) {
  case _FORTRAN_RUNTIME_IEEE_INVALID:
    return FE_INVALID;
  case _FORTRAN_RUNTIME_IEEE_DENORM:
    if (__FE_DENORM) {
      return __FE_DENORM;
    }
    break;
  case _FORTRAN_RUNTIME_IEEE_DIVIDE_BY_ZERO:
    return FE_DIVBYZERO;
  case _FORTRAN_RUNTIME_IEEE_OVERFLOW:
    return FE_OVERFLOW;
  case _FORTRAN_RUNTIME_IEEE_UNDERFLOW:
    return FE_UNDERFLOW;
  case _FORTRAN_RUNTIME_IEEE_INEXACT:
    return FE_INEXACT;
  }

  terminator.Crash("Invalid exception set: %d", except);
}

// Verify that the size of ieee_modes_type and ieee_status_type objects from
// intrinsic module file __fortran_ieee_exceptions.f90 are large enough to
// hold fenv_t object.
// TODO: fenv_t can be way larger than
//	sizeof(int) * _FORTRAN_RUNTIME_IEEE_FENV_T_EXTENT
// on some systems, e.g. Solaris, so omit object size comparison for now.
// TODO: consider femode_t object size comparison once its more mature.

} // extern "C"
} // namespace Fortran::runtime
