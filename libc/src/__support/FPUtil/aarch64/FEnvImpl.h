//===-- aarch64 floating point env manipulation functions -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_FPUTIL_AARCH64_FENVIMPL_H
#define LLVM_LIBC_SRC_SUPPORT_FPUTIL_AARCH64_FENVIMPL_H

#include "src/__support/common.h"
#include "src/__support/macros/properties/architectures.h"

#if !defined(LIBC_TARGET_ARCH_IS_AARCH64) || defined(__APPLE__)
#error "Invalid include"
#endif

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

  static constexpr uint32_t TONEAREST = 0x0;
  static constexpr uint32_t UPWARD = 0x1;
  static constexpr uint32_t DOWNWARD = 0x2;
  static constexpr uint32_t TOWARDZERO = 0x3;

  static constexpr uint32_t INVALID = 0x1;
  static constexpr uint32_t DIVBYZERO = 0x2;
  static constexpr uint32_t OVERFLOW = 0x4;
  static constexpr uint32_t UNDERFLOW = 0x8;
  static constexpr uint32_t INEXACT = 0x10;

  // Zero-th bit is the first bit.
  static constexpr uint32_t RoundingControlBitPosition = 22;
  static constexpr uint32_t ExceptionStatusFlagsBitPosition = 0;
  static constexpr uint32_t ExceptionControlFlagsBitPosition = 8;

  LIBC_INLINE static uint32_t getStatusValueForExcept(int excepts) {
    return (excepts & FE_INVALID ? INVALID : 0) |
           (excepts & FE_DIVBYZERO ? DIVBYZERO : 0) |
           (excepts & FE_OVERFLOW ? OVERFLOW : 0) |
           (excepts & FE_UNDERFLOW ? UNDERFLOW : 0) |
           (excepts & FE_INEXACT ? INEXACT : 0);
  }

  LIBC_INLINE static int exceptionStatusToMacro(uint32_t status) {
    return (status & INVALID ? FE_INVALID : 0) |
           (status & DIVBYZERO ? FE_DIVBYZERO : 0) |
           (status & OVERFLOW ? FE_OVERFLOW : 0) |
           (status & UNDERFLOW ? FE_UNDERFLOW : 0) |
           (status & INEXACT ? FE_INEXACT : 0);
  }

  static uint32_t getControlWord() { return __arm_rsr("fpcr"); }

  static void writeControlWord(uint32_t fpcr) { __arm_wsr("fpcr", fpcr); }

  static uint32_t getStatusWord() { return __arm_rsr("fpsr"); }

  static void writeStatusWord(uint32_t fpsr) { __arm_wsr("fpsr", fpsr); }
};

LIBC_INLINE int enable_except(int excepts) {
  uint32_t newExcepts = FEnv::getStatusValueForExcept(excepts);
  uint32_t controlWord = FEnv::getControlWord();
  int oldExcepts =
      (controlWord >> FEnv::ExceptionControlFlagsBitPosition) & 0x1F;
  controlWord |= (newExcepts << FEnv::ExceptionControlFlagsBitPosition);
  FEnv::writeControlWord(controlWord);
  return FEnv::exceptionStatusToMacro(oldExcepts);
}

LIBC_INLINE int disable_except(int excepts) {
  uint32_t disabledExcepts = FEnv::getStatusValueForExcept(excepts);
  uint32_t controlWord = FEnv::getControlWord();
  int oldExcepts =
      (controlWord >> FEnv::ExceptionControlFlagsBitPosition) & 0x1F;
  controlWord &= ~(disabledExcepts << FEnv::ExceptionControlFlagsBitPosition);
  FEnv::writeControlWord(controlWord);
  return FEnv::exceptionStatusToMacro(oldExcepts);
}

LIBC_INLINE int get_except() {
  uint32_t controlWord = FEnv::getControlWord();
  int enabledExcepts =
      (controlWord >> FEnv::ExceptionControlFlagsBitPosition) & 0x1F;
  return FEnv::exceptionStatusToMacro(enabledExcepts);
}

LIBC_INLINE int clear_except(int excepts) {
  uint32_t statusWord = FEnv::getStatusWord();
  uint32_t toClear = FEnv::getStatusValueForExcept(excepts);
  statusWord &= ~(toClear << FEnv::ExceptionStatusFlagsBitPosition);
  FEnv::writeStatusWord(statusWord);
  return 0;
}

LIBC_INLINE int test_except(int excepts) {
  uint32_t toTest = FEnv::getStatusValueForExcept(excepts);
  uint32_t statusWord = FEnv::getStatusWord();
  return FEnv::exceptionStatusToMacro(
      (statusWord >> FEnv::ExceptionStatusFlagsBitPosition) & toTest);
}

LIBC_INLINE int set_except(int excepts) {
  uint32_t statusWord = FEnv::getStatusWord();
  uint32_t statusValue = FEnv::getStatusValueForExcept(excepts);
  statusWord |= (statusValue << FEnv::ExceptionStatusFlagsBitPosition);
  FEnv::writeStatusWord(statusWord);
  return 0;
}

LIBC_INLINE int raise_except(int excepts) {
  float zero = 0.0f;
  float one = 1.0f;
  float largeValue = float(FPBits<float>(FPBits<float>::MAX_NORMAL));
  float smallValue = float(FPBits<float>(FPBits<float>::MIN_NORMAL));
  auto divfunc = [](float a, float b) {
    __asm__ __volatile__("ldr  s0, %0\n\t"
                         "ldr  s1, %1\n\t"
                         "fdiv s0, s0, s1\n\t"
                         : // No outputs
                         : "m"(a), "m"(b)
                         : "s0", "s1" /* s0 and s1 are clobbered */);
  };

  uint32_t toRaise = FEnv::getStatusValueForExcept(excepts);
  int result = 0;

  if (toRaise & FEnv::INVALID) {
    divfunc(zero, zero);
    uint32_t statusWord = FEnv::getStatusWord();
    if (!((statusWord >> FEnv::ExceptionStatusFlagsBitPosition) &
          FEnv::INVALID))
      result = -1;
  }

  if (toRaise & FEnv::DIVBYZERO) {
    divfunc(one, zero);
    uint32_t statusWord = FEnv::getStatusWord();
    if (!((statusWord >> FEnv::ExceptionStatusFlagsBitPosition) &
          FEnv::DIVBYZERO))
      result = -1;
  }
  if (toRaise & FEnv::OVERFLOW) {
    divfunc(largeValue, smallValue);
    uint32_t statusWord = FEnv::getStatusWord();
    if (!((statusWord >> FEnv::ExceptionStatusFlagsBitPosition) &
          FEnv::OVERFLOW))
      result = -1;
  }
  if (toRaise & FEnv::UNDERFLOW) {
    divfunc(smallValue, largeValue);
    uint32_t statusWord = FEnv::getStatusWord();
    if (!((statusWord >> FEnv::ExceptionStatusFlagsBitPosition) &
          FEnv::UNDERFLOW))
      result = -1;
  }
  if (toRaise & FEnv::INEXACT) {
    float two = 2.0f;
    float three = 3.0f;
    // 2.0 / 3.0 cannot be represented exactly in any radix 2 floating point
    // format.
    divfunc(two, three);
    uint32_t statusWord = FEnv::getStatusWord();
    if (!((statusWord >> FEnv::ExceptionStatusFlagsBitPosition) &
          FEnv::INEXACT))
      result = -1;
  }
  return result;
}

LIBC_INLINE int get_round() {
  uint32_t roundingMode =
      (FEnv::getControlWord() >> FEnv::RoundingControlBitPosition) & 0x3;
  switch (roundingMode) {
  case FEnv::TONEAREST:
    return FE_TONEAREST;
  case FEnv::DOWNWARD:
    return FE_DOWNWARD;
  case FEnv::UPWARD:
    return FE_UPWARD;
  case FEnv::TOWARDZERO:
    return FE_TOWARDZERO;
  default:
    return -1; // Error value.
  }
}

LIBC_INLINE int set_round(int mode) {
  uint16_t bitValue;
  switch (mode) {
  case FE_TONEAREST:
    bitValue = FEnv::TONEAREST;
    break;
  case FE_DOWNWARD:
    bitValue = FEnv::DOWNWARD;
    break;
  case FE_UPWARD:
    bitValue = FEnv::UPWARD;
    break;
  case FE_TOWARDZERO:
    bitValue = FEnv::TOWARDZERO;
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

LIBC_INLINE int get_env(fenv_t *envp) {
  FEnv::FPState *state = reinterpret_cast<FEnv::FPState *>(envp);
  state->ControlWord = FEnv::getControlWord();
  state->StatusWord = FEnv::getStatusWord();
  return 0;
}

LIBC_INLINE int set_env(const fenv_t *envp) {
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
