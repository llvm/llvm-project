//===-- x86_64 floating point env manipulation functions --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_FPUTIL_X86_64_FENV_H
#define LLVM_LIBC_UTILS_FPUTIL_X86_64_FENV_H

#include <fenv.h>
#include <stdint.h>
#include <xmmintrin.h>

namespace __llvm_libc {
namespace fputil {

namespace internal {

// Normally, one should be able to define FE_* macros to the exact rounding mode
// encodings. However, since we want LLVM libc to be compiled against headers
// from other libcs, we cannot assume that FE_* macros are always defined in
// such a manner. So, we will define enums corresponding to the x86_64 bit
// encodings. The implementations can map from FE_* to the corresponding enum
// values.

// The rounding control values in the x87 control register and the MXCSR
// register have the same 2-bit enoding but have different bit positions.
// See below for the bit positions.
struct RoundingControlValue {
  static constexpr uint16_t ToNearest = 0x0;
  static constexpr uint16_t Downward = 0x1;
  static constexpr uint16_t Upward = 0x2;
  static constexpr uint16_t TowardZero = 0x3;
};

static constexpr uint16_t X87RoundingControlBitPosition = 10;
static constexpr uint16_t MXCSRRoundingControlBitPosition = 13;

// The exception flags in the x87 status register and the MXCSR have the same
// encoding as well as the same bit positions.
struct ExceptionFlags {
  static constexpr uint16_t Invalid = 0x1;
  static constexpr uint16_t Denormal = 0x2; // This flag is not used
  static constexpr uint16_t DivByZero = 0x4;
  static constexpr uint16_t Overflow = 0x8;
  static constexpr uint16_t Underflow = 0x10;
  static constexpr uint16_t Inexact = 0x20;
};

// Exception flags are individual bits in the corresponding registers.
// So, we just OR the bit values to get the full set of exceptions.
static inline uint16_t getStatusValueForExcept(int excepts) {
  // We will make use of the fact that exception control bits are single
  // bit flags in the control registers.
  return (excepts & FE_INVALID ? ExceptionFlags::Invalid : 0) |
         (excepts & FE_DIVBYZERO ? ExceptionFlags::DivByZero : 0) |
         (excepts & FE_OVERFLOW ? ExceptionFlags::Overflow : 0) |
         (excepts & FE_UNDERFLOW ? ExceptionFlags::Underflow : 0) |
         (excepts & FE_INEXACT ? ExceptionFlags::Inexact : 0);
}

static inline int exceptionStatusToMacro(uint16_t status) {
  return (status & ExceptionFlags::Invalid ? FE_INVALID : 0) |
         (status & ExceptionFlags::DivByZero ? FE_DIVBYZERO : 0) |
         (status & ExceptionFlags::Overflow ? FE_OVERFLOW : 0) |
         (status & ExceptionFlags::Underflow ? FE_UNDERFLOW : 0) |
         (status & ExceptionFlags::Inexact ? FE_INEXACT : 0);
}

static inline uint16_t getX87ControlWord() {
  uint16_t w;
  __asm__ __volatile__("fnstcw %0" : "=m"(w)::);
  return w;
}

static inline void writeX87ControlWord(uint16_t w) {
  __asm__ __volatile__("fldcw %0" : : "m"(w) :);
}

static inline uint16_t getX87StatusWord() {
  uint16_t w;
  __asm__ __volatile__("fnstsw %0" : "=m"(w)::);
  return w;
}

static inline void clearX87Exceptions() {
  __asm__ __volatile__("fnclex" : : :);
}

} // namespace internal

static inline int clearExcept(int excepts) {
  // An instruction to write to x87 status word ins't available. So, we
  // just clear all of the x87 exceptions.
  // TODO: One can potentially use fegetenv/fesetenv to clear only the
  // listed exceptions in the x87 status word. We can do this if it is
  // really required.
  internal::clearX87Exceptions();

  uint32_t mxcsr = _mm_getcsr();
  mxcsr &= ~internal::getStatusValueForExcept(excepts);
  _mm_setcsr(mxcsr);
  return 0;
}

static inline int testExcept(int excepts) {
  uint16_t statusValue = internal::getStatusValueForExcept(excepts);
  // Check both x87 status word and MXCSR.
  return internal::exceptionStatusToMacro(
      (statusValue & internal::getX87StatusWord()) |
      (statusValue & _mm_getcsr()));
}

static inline int raiseExcept(int excepts) {
  // It is enough to set the exception flags in MXCSR.
  // TODO: Investigate if each exception has to be raised one at a time
  // followed with an fwait instruction before writing the flag for the
  // next exception.
  uint16_t statusValue = internal::getStatusValueForExcept(excepts);
  uint32_t sse = _mm_getcsr();
  sse = sse | statusValue;
  _mm_setcsr(sse);
  return 0;
}

static inline int getRound() {
  uint16_t bitValue =
      (_mm_getcsr() >> internal::MXCSRRoundingControlBitPosition) & 0x3;
  switch (bitValue) {
  case internal::RoundingControlValue::ToNearest:
    return FE_TONEAREST;
  case internal::RoundingControlValue::Downward:
    return FE_DOWNWARD;
  case internal::RoundingControlValue::Upward:
    return FE_UPWARD;
  case internal::RoundingControlValue::TowardZero:
    return FE_TOWARDZERO;
  default:
    return -1; // Error value.
  }
}

static inline int setRound(int mode) {
  uint16_t bitValue;
  switch (mode) {
  case FE_TONEAREST:
    bitValue = internal::RoundingControlValue::ToNearest;
    break;
  case FE_DOWNWARD:
    bitValue = internal::RoundingControlValue::Downward;
    break;
  case FE_UPWARD:
    bitValue = internal::RoundingControlValue::Upward;
    break;
  case FE_TOWARDZERO:
    bitValue = internal::RoundingControlValue::TowardZero;
    break;
  default:
    return 1; // To indicate failure
  }

  uint16_t x87Value = bitValue << internal::X87RoundingControlBitPosition;
  uint16_t x87Control = internal::getX87ControlWord();
  x87Control =
      (x87Control & ~(0x3 << internal::X87RoundingControlBitPosition)) |
      x87Value;
  internal::writeX87ControlWord(x87Control);

  uint32_t mxcsrValue = bitValue << internal::MXCSRRoundingControlBitPosition;
  uint32_t mxcsrControl = _mm_getcsr();
  mxcsrControl =
      (mxcsrControl & ~(0x3 << internal::MXCSRRoundingControlBitPosition)) |
      mxcsrValue;
  _mm_setcsr(mxcsrControl);

  return 0;
}

} // namespace fputil
} // namespace __llvm_libc

#endif // LLVM_LIBC_UTILS_FPUTIL_X86_64_FENV_H
