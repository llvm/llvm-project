//===-- lib/runtime/exceptions.cpp ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Runtime exception support.

#include "flang/Runtime/exceptions.h"
#include "flang-rt/runtime/terminator.h"
#include "flang/Common/fp-control.h"
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
uint32_t RTDEF(MapException)(uint32_t excepts) {
  Terminator terminator{__FILE__, __LINE__};

#if defined(RT_DEVICE_COMPILATION)
  terminator.Crash(
      "not implemented yet: raising IEEE FP exception in device code: %d",
      excepts);
#else
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
#endif
}

// The following exception processing routines have a libm call component,
// and where available, an additional component for handling the nonstandard
// ieee_denorm exception. The denorm component does not subsume the libm
// component; both are needed.

void RTNAME(feclearexcept)(uint32_t excepts) {
  FLANG_FP_TRAP_ON
  feclearexcept(excepts);
#if defined(_MM_EXCEPT_DENORM)
  _mm_setcsr(_mm_getcsr() & ~(excepts & _MM_EXCEPT_MASK));
#endif
}
void RTDEF(feraiseexcept)(uint32_t excepts) {
#if !defined(RT_DEVICE_COMPILATION)
  FLANG_FP_TRAP_ON
  feraiseexcept(excepts);
#if defined(_MM_EXCEPT_DENORM)
  _mm_setcsr(_mm_getcsr() | (excepts & _MM_EXCEPT_MASK));
#endif
#endif
}
uint32_t RTNAME(fetestexcept)(uint32_t excepts) {
  FLANG_FP_TRAP_ON
#if defined(_MM_EXCEPT_DENORM)
  return (_mm_getcsr() & _MM_EXCEPT_MASK & excepts) | fetestexcept(excepts);
#else
  return fetestexcept(excepts);
#endif
}
void RTNAME(fedisableexcept)(uint32_t excepts) {
#ifdef __USE_GNU
  fedisableexcept(excepts);
#endif
#if defined(_MM_EXCEPT_DENORM)
  _mm_setcsr(_mm_getcsr() | ((excepts & _MM_EXCEPT_MASK) << 7));
#endif
}
void RTNAME(feenableexcept)(uint32_t excepts) {
#ifdef __USE_GNU
  feenableexcept(excepts);
#endif
#if defined(_MM_EXCEPT_DENORM)
  _mm_setcsr(_mm_getcsr() & ~((excepts & _MM_EXCEPT_MASK) << 7));
#endif
}
uint32_t RTNAME(fegetexcept)() {
  uint32_t excepts = 0;
#ifdef __USE_GNU
  excepts = fegetexcept();
#endif
#if defined(_MM_EXCEPT_DENORM)
  return (63 - ((_mm_getcsr() >> 7) & _MM_EXCEPT_MASK)) | excepts;
#else
  return excepts;
#endif
}

// Check if the processor has the ability to control whether to halt or
// continue execution when a given exception is raised.
//
// TODO: Support halting on x86 without glibc. The MXCSR helpers above (guarded
// by _MM_EXCEPT_DENORM) already provide the machinery, so this gate could be
// widened to `#if defined(__USE_GNU) || defined(_MM_EXCEPT_DENORM)` to cover
// x86_64 macOS/BSD and musl-Linux. Doing so also needs the x87 control word for
// REAL(10), 32-bit x86 (__i386__), and Windows SEH handling; see PR discussion.
bool RTNAME(SupportHalting)([[maybe_unused]] uint32_t except) {
#ifdef __USE_GNU
  except = RTNAME(MapException)(except);
  int currentSet = RTNAME(fegetexcept)(), flipSet;
  if (currentSet & except) {
    RTNAME(fedisableexcept)(except);
    flipSet = RTNAME(fegetexcept)();
    RTNAME(feenableexcept)(except);
  } else {
    RTNAME(feenableexcept)(except);
    flipSet = RTNAME(fegetexcept)();
    RTNAME(fedisableexcept)(except);
  }
  return currentSet != flipSet;
#else
  return false;
#endif
}

void RTNAME(EnableFPETraps)(uint32_t excepts) {
  // Enable halting only for those exceptions whose halting control is supported
  // by the processor. The Fortran standard restricts IEEE_SET_HALTING_MODE to
  // flags for which IEEE_SUPPORT_HALTING is true (F2023 17.11.40); on targets
  // without halting control (e.g. non-glibc), this is a no-op.
  uint32_t supported = 0;
  for (uint32_t flag = 1; flag <= excepts; flag <<= 1) {
    if ((excepts & flag) && RTNAME(SupportHalting)(flag)) {
      supported |= flag;
    }
  }
  if (supported == 0) {
    return;
  }
  uint32_t mapped = RTNAME(MapException)(supported);
  RTNAME(feclearexcept)(mapped);
  RTNAME(feenableexcept)(mapped);
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
