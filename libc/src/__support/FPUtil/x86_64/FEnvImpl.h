//===-- x86_64 floating point env manipulation functions --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_FPUTIL_X86_64_FENVIMPL_H
#define LLVM_LIBC_SRC_SUPPORT_FPUTIL_X86_64_FENVIMPL_H

#include <fenv.h>
#include <stdint.h>

#include "src/__support/sanitizer.h"

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
  // Some libcs define __FE_DENORM corresponding to the denormal input
  // exception and include it in FE_ALL_EXCEPTS. We define and use it to
  // support compiling against headers provided by such libcs.
  static constexpr uint16_t Denormal = 0x2;
  static constexpr uint16_t DivByZero = 0x4;
  static constexpr uint16_t Overflow = 0x8;
  static constexpr uint16_t Underflow = 0x10;
  static constexpr uint16_t Inexact = 0x20;
};

// The exception control bits occupy six bits, one bit for each exception.
// In the x87 control word, they occupy the first 6 bits. In the MXCSR
// register, they occupy bits 7 to 12.
static constexpr uint16_t X87ExceptionControlBitPosition = 0;
static constexpr uint16_t MXCSRExceptionContolBitPoistion = 7;

// Exception flags are individual bits in the corresponding registers.
// So, we just OR the bit values to get the full set of exceptions.
static inline uint16_t getStatusValueForExcept(int excepts) {
  // We will make use of the fact that exception control bits are single
  // bit flags in the control registers.
  return (excepts & FE_INVALID ? ExceptionFlags::Invalid : 0) |
#ifdef __FE_DENORM
         (excepts & __FE_DENORM ? ExceptionFlags::Denormal : 0) |
#endif // __FE_DENORM
         (excepts & FE_DIVBYZERO ? ExceptionFlags::DivByZero : 0) |
         (excepts & FE_OVERFLOW ? ExceptionFlags::Overflow : 0) |
         (excepts & FE_UNDERFLOW ? ExceptionFlags::Underflow : 0) |
         (excepts & FE_INEXACT ? ExceptionFlags::Inexact : 0);
}

static inline int exceptionStatusToMacro(uint16_t status) {
  return (status & ExceptionFlags::Invalid ? FE_INVALID : 0) |
#ifdef __FE_DENORM
         (status & ExceptionFlags::Denormal ? __FE_DENORM : 0) |
#endif // __FE_DENORM
         (status & ExceptionFlags::DivByZero ? FE_DIVBYZERO : 0) |
         (status & ExceptionFlags::Overflow ? FE_OVERFLOW : 0) |
         (status & ExceptionFlags::Underflow ? FE_UNDERFLOW : 0) |
         (status & ExceptionFlags::Inexact ? FE_INEXACT : 0);
}

struct X87StateDescriptor {
  uint16_t ControlWord;
  uint16_t Unused1;
  uint16_t StatusWord;
  uint16_t Unused2;
  // TODO: Elaborate the remaining 20 bytes as required.
  uint32_t _[5];
};

static inline uint16_t getX87ControlWord() {
  uint16_t w;
  __asm__ __volatile__("fnstcw %0" : "=m"(w)::);
  SANITIZER_MEMORY_INITIALIZED(&w, sizeof(w));
  return w;
}

static inline void writeX87ControlWord(uint16_t w) {
  __asm__ __volatile__("fldcw %0" : : "m"(w) :);
}

static inline uint16_t getX87StatusWord() {
  uint16_t w;
  __asm__ __volatile__("fnstsw %0" : "=m"(w)::);
  SANITIZER_MEMORY_INITIALIZED(&w, sizeof(w));
  return w;
}

static inline void clearX87Exceptions() {
  __asm__ __volatile__("fnclex" : : :);
}

static inline uint32_t getMXCSR() {
  uint32_t w;
  __asm__ __volatile__("stmxcsr %0" : "=m"(w)::);
  SANITIZER_MEMORY_INITIALIZED(&w, sizeof(w));
  return w;
}

static inline void writeMXCSR(uint32_t w) {
  __asm__ __volatile__("ldmxcsr %0" : : "m"(w) :);
}

static inline void getX87StateDescriptor(X87StateDescriptor &s) {
  __asm__ __volatile__("fnstenv %0" : "=m"(s));
  SANITIZER_MEMORY_INITIALIZED(&s, sizeof(s));
}

static inline void writeX87StateDescriptor(const X87StateDescriptor &s) {
  __asm__ __volatile__("fldenv %0" : : "m"(s) :);
}

static inline void fwait() { __asm__ __volatile__("fwait"); }

} // namespace internal

static inline int enableExcept(int excepts) {
  // In the x87 control word and in MXCSR, an exception is blocked
  // if the corresponding bit is set. That is the reason for all the
  // bit-flip operations below as we need to turn the bits to zero
  // to enable them.

  uint16_t bitMask = internal::getStatusValueForExcept(excepts);

  uint16_t x87CW = internal::getX87ControlWord();
  uint16_t oldExcepts = ~x87CW & 0x3F; // Save previously enabled exceptions.
  x87CW &= ~bitMask;
  internal::writeX87ControlWord(x87CW);

  // Enabling SSE exceptions via MXCSR is a nice thing to do but
  // might not be of much use practically as SSE exceptions and the x87
  // exceptions are independent of each other.
  uint32_t mxcsr = internal::getMXCSR();
  mxcsr &= ~(bitMask << internal::MXCSRExceptionContolBitPoistion);
  internal::writeMXCSR(mxcsr);

  // Since the x87 exceptions and SSE exceptions are independent of each,
  // it doesn't make much sence to report both in the return value. Most
  // often, the standard floating point functions deal with FPU operations
  // so we will retrun only the old x87 exceptions.
  return internal::exceptionStatusToMacro(oldExcepts);
}

static inline int disableExcept(int excepts) {
  // In the x87 control word and in MXCSR, an exception is blocked
  // if the corresponding bit is set.

  uint16_t bitMask = internal::getStatusValueForExcept(excepts);

  uint16_t x87CW = internal::getX87ControlWord();
  uint16_t oldExcepts = ~x87CW & 0x3F; // Save previously enabled exceptions.
  x87CW |= bitMask;
  internal::writeX87ControlWord(x87CW);

  // Just like in enableExcept, it is not clear if disabling SSE exceptions
  // is required. But, we will still do it only as a "nice thing to do".
  uint32_t mxcsr = internal::getMXCSR();
  mxcsr |= (bitMask << internal::MXCSRExceptionContolBitPoistion);
  internal::writeMXCSR(mxcsr);

  return internal::exceptionStatusToMacro(oldExcepts);
}

static inline int clearExcept(int excepts) {
  internal::X87StateDescriptor state;
  internal::getX87StateDescriptor(state);
  state.StatusWord &= ~internal::getStatusValueForExcept(excepts);
  internal::writeX87StateDescriptor(state);

  uint32_t mxcsr = internal::getMXCSR();
  mxcsr &= ~internal::getStatusValueForExcept(excepts);
  internal::writeMXCSR(mxcsr);
  return 0;
}

static inline int testExcept(int excepts) {
  uint16_t statusValue = internal::getStatusValueForExcept(excepts);
  // Check both x87 status word and MXCSR.
  return internal::exceptionStatusToMacro(
      (statusValue & internal::getX87StatusWord()) |
      (statusValue & internal::getMXCSR()));
}

// Sets the exception flags but does not trigger the exception handler.
static inline int setExcept(int excepts) {
  uint16_t statusValue = internal::getStatusValueForExcept(excepts);
  internal::X87StateDescriptor state;
  internal::getX87StateDescriptor(state);
  state.StatusWord |= statusValue;
  internal::writeX87StateDescriptor(state);

  uint32_t mxcsr = internal::getMXCSR();
  mxcsr |= statusValue;
  internal::writeMXCSR(mxcsr);

  return 0;
}

static inline int raiseExcept(int excepts) {
  uint16_t statusValue = internal::getStatusValueForExcept(excepts);

  // We set the status flag for exception one at a time and call the
  // fwait instruction to actually get the processor to raise the
  // exception by calling the exception handler. This scheme is per
  // the description in in "8.6 X87 FPU EXCEPTION SYNCHRONIZATION"
  // of the "Intel 64 and IA-32 Architectures Software Developer's
  // Manual, Vol 1".

  // FPU status word is read for each exception seperately as the
  // exception handler can potentially write to it (typically to clear
  // the corresponding exception flag). By reading it separately, we
  // ensure that the writes by the exception handler are maintained
  // when raising the next exception.

  auto raiseHelper = [](uint16_t singleExceptFlag) {
    internal::X87StateDescriptor state;
    internal::getX87StateDescriptor(state);
    state.StatusWord |= singleExceptFlag;
    internal::writeX87StateDescriptor(state);
    internal::fwait();
  };

  if (statusValue & internal::ExceptionFlags::Invalid)
    raiseHelper(internal::ExceptionFlags::Invalid);
  if (statusValue & internal::ExceptionFlags::DivByZero)
    raiseHelper(internal::ExceptionFlags::DivByZero);
  if (statusValue & internal::ExceptionFlags::Overflow)
    raiseHelper(internal::ExceptionFlags::Overflow);
  if (statusValue & internal::ExceptionFlags::Underflow)
    raiseHelper(internal::ExceptionFlags::Underflow);
  if (statusValue & internal::ExceptionFlags::Inexact)
    raiseHelper(internal::ExceptionFlags::Inexact);
#ifdef __FE_DENORM
  if (statusValue & internal::ExceptionFlags::Denormal) {
    raiseHelper(internal::ExceptionFlags::Denormal);
  }
#endif // __FE_DENORM

  // There is no special synchronization scheme available to
  // raise SEE exceptions. So, we will ignore that for now.
  // Just plain writing to the MXCSR register does not guarantee
  // the exception handler will be called.

  return 0;
}

static inline int getRound() {
  uint16_t bitValue =
      (internal::getMXCSR() >> internal::MXCSRRoundingControlBitPosition) & 0x3;
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

  uint16_t x87Value = static_cast<uint16_t>(
      bitValue << internal::X87RoundingControlBitPosition);
  uint16_t x87Control = internal::getX87ControlWord();
  x87Control = static_cast<uint16_t>(
      (x87Control &
       ~(uint16_t(0x3) << internal::X87RoundingControlBitPosition)) |
      x87Value);
  internal::writeX87ControlWord(x87Control);

  uint32_t mxcsrValue = bitValue << internal::MXCSRRoundingControlBitPosition;
  uint32_t mxcsrControl = internal::getMXCSR();
  mxcsrControl =
      (mxcsrControl & ~(0x3 << internal::MXCSRRoundingControlBitPosition)) |
      mxcsrValue;
  internal::writeMXCSR(mxcsrControl);

  return 0;
}

namespace internal {

#ifdef _WIN32
// MSVC fenv.h defines a very simple representation of the floating point state
// which just consists of control and status words of the x87 unit.
struct FPState {
  uint32_t ControlWord;
  uint32_t StatusWord;
};
#else
struct FPState {
  X87StateDescriptor X87Status;
  uint32_t MXCSR;
};
#endif // _WIN32

} // namespace internal

static_assert(
    sizeof(fenv_t) == sizeof(internal::FPState),
    "Internal floating point state does not match the public fenv_t type.");

#ifdef _WIN32
static inline int getEnv(fenv_t *envp) {
  internal::FPState *state = reinterpret_cast<internal::FPState *>(envp);
  internal::X87StateDescriptor X87Status;
  internal::getX87StateDescriptor(X87Status);
  state->ControlWord = X87Status.ControlWord;
  state->StatusWord = X87Status.StatusWord;
  return 0;
}

static inline int setEnv(const fenv_t *envp) {
  const internal::FPState *state =
      reinterpret_cast<const internal::FPState *>(envp);
  internal::X87StateDescriptor X87Status;
  X87Status.ControlWord = state->ControlWord;
  X87Status.StatusWord = state->StatusWord;
  internal::writeX87StateDescriptor(X87Status);
  return 0;
}
#else
static inline int getEnv(fenv_t *envp) {
  internal::FPState *state = reinterpret_cast<internal::FPState *>(envp);
  internal::getX87StateDescriptor(state->X87Status);
  state->MXCSR = internal::getMXCSR();
  return 0;
}

static inline int setEnv(const fenv_t *envp) {
  const internal::FPState *state =
      reinterpret_cast<const internal::FPState *>(envp);
  internal::writeX87StateDescriptor(state->X87Status);
  internal::writeMXCSR(state->MXCSR);
  return 0;
}
#endif

} // namespace fputil
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_FPUTIL_X86_64_FENVIMPL_H
