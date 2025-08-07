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
#if defined(__aarch64__) && defined(__GLIBC__)
#include <fpu_control.h>
#elif defined(__x86_64__) && !defined(_WIN32)
#include <xmmintrin.h>
#endif

// File fenv.h usually, but not always, defines standard exceptions as both
// enumerator values and preprocessor #defines. Some x86 environments also
// define a nonstandard __FE_DENORM enumerator, but without a corresponding
// #define, which makes it more difficult to determine if it is present or not.
#ifndef FE_INVALID
#define FE_INVALID 0
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
#if FE_INVALID == 1 && FE_DIVBYZERO == 4 && FE_OVERFLOW == 8 && \
    FE_UNDERFLOW == 16 && FE_INEXACT == 32
#define __FE_DENORM 2
#else
#define __FE_DENORM 0
#endif

namespace Fortran::runtime {

extern "C" {

// Map a set of Fortran ieee_arithmetic module exceptions to a libm fenv.h
// excepts value.
uint32_t RTNAME(MapException)(uint32_t excepts) {
  Terminator terminator{__FILE__, __LINE__};

  static constexpr uint32_t v{FE_INVALID};
  static constexpr uint32_t s{__FE_DENORM};
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
  if (excepts >= mapSize) {
    terminator.Crash("Invalid excepts value: %d", excepts);
  }
  uint32_t except_value = map[excepts];
  return except_value;
}

// Check if the processor has the ability to control whether to halt or
// continue execution when a given exception is raised.
bool RTNAME(SupportHalting)([[maybe_unused]] uint32_t except) {
#ifdef __USE_GNU
  except = RTNAME(MapException)(except);
  int currentSet = fegetexcept(), flipSet, ok;
  if (currentSet & except) {
    ok = fedisableexcept(except);
    flipSet = fegetexcept();
    ok |= feenableexcept(except);
  } else {
    ok = feenableexcept(except);
    flipSet = fegetexcept();
    ok |= fedisableexcept(except);
  }
  return ok != -1 && currentSet != flipSet;
#else
  return false;
#endif
}

// A hardware FZ (flush to zero) bit is the negation of the
// ieee_[get|set]_underflow_mode GRADUAL argument.
#if defined(_MM_FLUSH_ZERO_MASK)
// The x86_64 MXCSR FZ bit affects computations of real kinds 3, 4, and 8.
#elif defined(_FPU_GETCW)
// The aarch64 FPCR FZ bit affects computations of real kinds 3, 4, and 8.
// bit 24: FZ   -- single, double precision flush to zero bit
// bit 19: FZ16 -- half precision flush to zero bit [not currently relevant]
#define _FPU_FPCR_FZ_MASK_ 0x01080000
#endif

bool RTNAME(GetUnderflowMode)(void) {
#if defined(_MM_FLUSH_ZERO_MASK)
  return _MM_GET_FLUSH_ZERO_MODE() == _MM_FLUSH_ZERO_OFF;
#elif defined(_FPU_GETCW)
  uint64_t fpcr;
  _FPU_GETCW(fpcr);
  return (fpcr & _FPU_FPCR_FZ_MASK_) == 0;
#else
  return false;
#endif
}
void RTNAME(SetUnderflowMode)(bool flag) {
#if defined(_MM_FLUSH_ZERO_MASK)
  _MM_SET_FLUSH_ZERO_MODE(flag ? _MM_FLUSH_ZERO_OFF : _MM_FLUSH_ZERO_ON);
#elif defined(_FPU_GETCW)
  uint64_t fpcr;
  _FPU_GETCW(fpcr);
  if (flag) {
    fpcr &= ~_FPU_FPCR_FZ_MASK_;
  } else {
    fpcr |= _FPU_FPCR_FZ_MASK_;
  }
  _FPU_SETCW(fpcr);
#endif
}

size_t RTNAME(GetModesTypeSize)(void) {
#ifdef __GLIBC_USE_IEC_60559_BFP_EXT
  return sizeof(femode_t); // byte size of ieee_modes_type data
#else
  return 8; // femode_t is not defined
#endif
}
size_t RTNAME(GetStatusTypeSize)(void) {
  return sizeof(fenv_t); // byte size of ieee_status_type data
}

} // extern "C"
} // namespace Fortran::runtime
