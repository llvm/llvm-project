//===-- aarch64 floating point env manipulation functions -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_FPUTIL_AARCH64_FENVIMPL_H
#define LLVM_LIBC_SRC_SUPPORT_FPUTIL_AARCH64_FENVIMPL_H

#include <arm_acle.h>
#include <fenv.h>
#include <stdint.h>

#include "src/__support/FPUtil/FPBits.h"

namespace __llvm_libc {
namespace fputil {

struct FEnv {
  struct FPState {
    uint32_t ControlWord;
    uint32_t StatusWord;
  };

  static_assert(
      sizeof(fenv_t) == sizeof(FPState),
      "Internal floating point state does not match the public fenv_t type.");

  static constexpr uint32_t ToNearest = 0x0;
  static constexpr uint32_t Upward = 0x1;
  static constexpr uint32_t Downward = 0x2;
  static constexpr uint32_t TowardZero = 0x3;

  static constexpr uint32_t Invalid = 0x1;
  static constexpr uint32_t DivByZero = 0x2;
  static constexpr uint32_t Overflow = 0x4;
  static constexpr uint32_t Underflow = 0x8;
  static constexpr uint32_t Inexact = 0x10;

  // Zero-th bit is the first bit.
  static constexpr uint32_t RoundingControlBitPosition = 22;
  static constexpr uint32_t ExceptionStatusFlagsBitPosition = 0;
  static constexpr uint32_t ExceptionControlFlagsBitPosition = 8;

  static inline uint32_t getStatusValueForExcept(int excepts) {
    return (excepts & FE_INVALID ? Invalid : 0) |
           (excepts & FE_DIVBYZERO ? DivByZero : 0) |
           (excepts & FE_OVERFLOW ? Overflow : 0) |
           (excepts & FE_UNDERFLOW ? Underflow : 0) |
           (excepts & FE_INEXACT ? Inexact : 0);
  }

  static inline int exceptionStatusToMacro(uint32_t status) {
    return (status & Invalid ? FE_INVALID : 0) |
           (status & DivByZero ? FE_DIVBYZERO : 0) |
           (status & Overflow ? FE_OVERFLOW : 0) |
           (status & Underflow ? FE_UNDERFLOW : 0) |
           (status & Inexact ? FE_INEXACT : 0);
  }

  static uint32_t getControlWord() { return __arm_rsr("fpcr"); }

  static void writeControlWord(uint32_t fpcr) { __arm_wsr("fpcr", fpcr); }

  static uint32_t getStatusWord() { return __arm_rsr("fpsr"); }

  static void writeStatusWord(uint32_t fpsr) { __arm_wsr("fpsr", fpsr); }
};

static inline int enableExcept(int excepts) {
  uint32_t newExcepts = FEnv::getStatusValueForExcept(excepts);
  uint32_t controlWord = FEnv::getControlWord();
  int oldExcepts =
      (controlWord >> FEnv::ExceptionControlFlagsBitPosition) & 0x1F;
  controlWord |= (newExcepts << FEnv::ExceptionControlFlagsBitPosition);
  FEnv::writeControlWord(controlWord);
  return FEnv::exceptionStatusToMacro(oldExcepts);
}

static inline int disableExcept(int excepts) {
  uint32_t disabledExcepts = FEnv::getStatusValueForExcept(excepts);
  uint32_t controlWord = FEnv::getControlWord();
  int oldExcepts =
      (controlWord >> FEnv::ExceptionControlFlagsBitPosition) & 0x1F;
  controlWord &= ~(disabledExcepts << FEnv::ExceptionControlFlagsBitPosition);
  FEnv::writeControlWord(controlWord);
  return FEnv::exceptionStatusToMacro(oldExcepts);
}

static inline int getExcept() {
  uint32_t controlWord = FEnv::getControlWord();
  int enabledExcepts =
      (controlWord >> FEnv::ExceptionControlFlagsBitPosition) & 0x1F;
  return FEnv::exceptionStatusToMacro(enabledExcepts);
}

static inline int clearExcept(int excepts) {
  uint32_t statusWord = FEnv::getStatusWord();
  uint32_t toClear = FEnv::getStatusValueForExcept(excepts);
  statusWord &= ~(toClear << FEnv::ExceptionStatusFlagsBitPosition);
  FEnv::writeStatusWord(statusWord);
  return 0;
}

static inline int testExcept(int excepts) {
  uint32_t toTest = FEnv::getStatusValueForExcept(excepts);
  uint32_t statusWord = FEnv::getStatusWord();
  return FEnv::exceptionStatusToMacro(
      (statusWord >> FEnv::ExceptionStatusFlagsBitPosition) & toTest);
}

static inline int setExcept(int excepts) {
  uint32_t statusWord = FEnv::getControlWord();
  uint32_t statusValue = FEnv::getStatusValueForExcept(excepts);
  statusWord |= (statusValue << FEnv::ExceptionStatusFlagsBitPosition);
  FEnv::writeStatusWord(statusWord);
  return 0;
}

static inline int raiseExcept(int excepts) {
  float zero = 0.0f;
  float one = 1.0f;
  float largeValue = float(FPBits<float>(FPBits<float>::maxNormal));
  float smallValue = float(FPBits<float>(FPBits<float>::minNormal));
  auto divfunc = [](float a, float b) {
    __asm__ __volatile__("ldr  s0, %0\n\t"
                         "ldr  s1, %1\n\t"
                         "fdiv s0, s0, s1\n\t"
                         : // No outputs
                         : "m"(a), "m"(b)
                         : "s0", "s1" /* s0 and s1 are clobbered */);
  };

  uint32_t toRaise = FEnv::getStatusValueForExcept(excepts);

  if (toRaise & FEnv::Invalid) {
    divfunc(zero, zero);
    uint32_t statusWord = FEnv::getStatusWord();
    if (!((statusWord >> FEnv::ExceptionStatusFlagsBitPosition) &
          FEnv::Invalid))
      return -1;
  }

  if (toRaise & FEnv::DivByZero) {
    divfunc(one, zero);
    uint32_t statusWord = FEnv::getStatusWord();
    if (!((statusWord >> FEnv::ExceptionStatusFlagsBitPosition) &
          FEnv::DivByZero))
      return -1;
  }
  if (toRaise & FEnv::Overflow) {
    divfunc(largeValue, smallValue);
    uint32_t statusWord = FEnv::getStatusWord();
    if (!((statusWord >> FEnv::ExceptionStatusFlagsBitPosition) &
          FEnv::Overflow))
      return -1;
  }
  if (toRaise & FEnv::Underflow) {
    divfunc(smallValue, largeValue);
    uint32_t statusWord = FEnv::getStatusWord();
    if (!((statusWord >> FEnv::ExceptionStatusFlagsBitPosition) &
          FEnv::Underflow))
      return -1;
  }
  if (toRaise & FEnv::Inexact) {
    float two = 2.0f;
    float three = 3.0f;
    // 2.0 / 3.0 cannot be represented exactly in any radix 2 floating point
    // format.
    divfunc(two, three);
    uint32_t statusWord = FEnv::getStatusWord();
    if (!((statusWord >> FEnv::ExceptionStatusFlagsBitPosition) &
          FEnv::Inexact))
      return -1;
  }
  return 0;
}

static inline int getRound() {
  uint32_t roundingMode =
      (FEnv::getControlWord() >> FEnv::RoundingControlBitPosition) & 0x3;
  switch (roundingMode) {
  case FEnv::ToNearest:
    return FE_TONEAREST;
  case FEnv::Downward:
    return FE_DOWNWARD;
  case FEnv::Upward:
    return FE_UPWARD;
  case FEnv::TowardZero:
    return FE_TOWARDZERO;
  default:
    return -1; // Error value.
  }
}

static inline int setRound(int mode) {
  uint16_t bitValue;
  switch (mode) {
  case FE_TONEAREST:
    bitValue = FEnv::ToNearest;
    break;
  case FE_DOWNWARD:
    bitValue = FEnv::Downward;
    break;
  case FE_UPWARD:
    bitValue = FEnv::Upward;
    break;
  case FE_TOWARDZERO:
    bitValue = FEnv::TowardZero;
    break;
  default:
    return 1; // To indicate failure
  }

  uint32_t controlWord = FEnv::getControlWord();
  controlWord &= ~(0x3 << FEnv::RoundingControlBitPosition);
  controlWord |= (bitValue << FEnv::RoundingControlBitPosition);
  FEnv::writeControlWord(controlWord);

  return 0;
}

static inline int getEnv(fenv_t *envp) {
  FEnv::FPState *state = reinterpret_cast<FEnv::FPState *>(envp);
  state->ControlWord = FEnv::getControlWord();
  state->StatusWord = FEnv::getStatusWord();
  return 0;
}

static inline int setEnv(const fenv_t *envp) {
  if (envp == FE_DFL_ENV) {
    // Default status and control words bits are all zeros so we just
    // write zeros.
    FEnv::writeStatusWord(0);
    FEnv::writeControlWord(0);
    return 0;
  }
  const FEnv::FPState *state = reinterpret_cast<const FEnv::FPState *>(envp);
  FEnv::writeControlWord(state->ControlWord);
  FEnv::writeStatusWord(state->StatusWord);
  return 0;
}

} // namespace fputil
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_FPUTIL_AARCH64_FENVIMPL_H
