//===-- runtime/exceptions.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Runtime exception support.

#include "flang/Runtime/exceptions.h"
#include "terminator.h"
#include <cfenv>

// When not supported, these macro are undefined in cfenv.h,
// set them to zero in that case.
#ifndef FE_INVALID
#define FE_INVALID 0
#endif
#ifndef __FE_DENORM
#define __FE_DENORM 0 // denorm is nonstandard
#endif
#ifndef FE_DIVBYZERO
#define FE_DIVBYZERO 0
#endif
#ifndef FE_OVERFLOW
#define FE_OVERFLOW 0
#endif
#ifndef FE_UNDERFLOW
#define FE_UNDERFLOW 0
#endif
#ifndef FE_INEXACT
#define FE_INEXACT 0
#endif

namespace Fortran::runtime {

extern "C" {

// Map a set of Fortran ieee_arithmetic module exceptions to a libm fenv.h
// excepts value.
uint32_t RTNAME(MapException)(uint32_t excepts) {
  Terminator terminator{__FILE__, __LINE__};

  static constexpr uint32_t v{FE_INVALID};
  static constexpr uint32_t s{__FE_DENORM}; // subnormal
  static constexpr uint32_t z{FE_DIVBYZERO};
  static constexpr uint32_t o{FE_OVERFLOW};
  static constexpr uint32_t u{FE_UNDERFLOW};
  static constexpr uint32_t x{FE_INEXACT};

#define vm(p) p, p | v
#define sm(p) vm(p), vm(p | s)
#define zm(p) sm(p), sm(p | z)
#define om(p) zm(p), zm(p | o)
#define um(p) om(p), om(p | u)
#define xm um(0), um(x)

  static constexpr uint32_t map[]{xm};
  static constexpr uint32_t mapSize{sizeof(map) / sizeof(uint32_t)};
  static_assert(mapSize == 64);
  if (excepts == 0 || excepts >= mapSize) {
    terminator.Crash("Invalid excepts value: %d", excepts);
  }
  uint32_t except_value = map[excepts];
  if (except_value == 0) {
    terminator.Crash(
        "Excepts value %d not supported by flang runtime", excepts);
  }
  return except_value;
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
