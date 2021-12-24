//===-- x86_64 floating point env manipulation functions --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_FPUTIL_X86_64_FENVIMPL_H
#define LLVM_LIBC_SRC_SUPPORT_FPUTIL_X86_64_FENVIMPL_H

#include "src/__support/architectures.h"

#if !defined(LLVM_LIBC_ARCH_X86)
#error "Invalid include"
#endif

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
  static constexpr uint16_t TO_NEAREST = 0x0;
  static constexpr uint16_t DOWNWARD = 0x1;
  static constexpr uint16_t UPWARD = 0x2;
  static constexpr uint16_t TOWARD_ZERO = 0x3;
};

static constexpr uint16_t X87_ROUNDING_CONTROL_BIT_POSITION = 10;
static constexpr uint16_t MXCSR_ROUNDING_CONTROL_BIT_POSITION = 13;

// The exception flags in the x87 status register and the MXCSR have the same
// encoding as well as the same bit positions.
struct ExceptionFlags {
  static constexpr uint16_t INVALID = 0x1;
  // Some libcs define __FE_DENORM corresponding to the denormal input
  // exception and include it in FE_ALL_EXCEPTS. We define and use it to
  // support compiling against headers provided by such libcs.
  static constexpr uint16_t DENORMAL = 0x2;
  static constexpr uint16_t DIV_BY_ZERO = 0x4;
  static constexpr uint16_t OVERFLOW = 0x8;
  static constexpr uint16_t UNDERFLOW = 0x10;
  static constexpr uint16_t INEXACT = 0x20;
};

// The exception control bits occupy six bits, one bit for each exception.
// In the x87 control word, they occupy the first 6 bits. In the MXCSR
// register, they occupy bits 7 to 12.
static constexpr uint16_t X87_EXCEPTION_CONTROL_BIT_POSITION = 0;
static constexpr uint16_t MXCSR_EXCEPTION_CONTOL_BIT_POISTION = 7;

// Exception flags are individual bits in the corresponding registers.
// So, we just OR the bit values to get the full set of exceptions.
static inline uint16_t get_status_value_for_except(int excepts) {
  // We will make use of the fact that exception control bits are single
  // bit flags in the control registers.
  return (excepts & FE_INVALID ? ExceptionFlags::INVALID : 0) |
#ifdef __FE_DENORM
         (excepts & __FE_DENORM ? ExceptionFlags::Denormal : 0) |
#endif // __FE_DENORM
         (excepts & FE_DIVBYZERO ? ExceptionFlags::DIV_BY_ZERO : 0) |
         (excepts & FE_OVERFLOW ? ExceptionFlags::OVERFLOW : 0) |
         (excepts & FE_UNDERFLOW ? ExceptionFlags::UNDERFLOW : 0) |
         (excepts & FE_INEXACT ? ExceptionFlags::INEXACT : 0);
}

static inline int exception_status_to_macro(uint16_t status) {
  return (status & ExceptionFlags::INVALID ? FE_INVALID : 0) |
#ifdef __FE_DENORM
         (status & ExceptionFlags::Denormal ? __FE_DENORM : 0) |
#endif // __FE_DENORM
         (status & ExceptionFlags::DIV_BY_ZERO ? FE_DIVBYZERO : 0) |
         (status & ExceptionFlags::OVERFLOW ? FE_OVERFLOW : 0) |
         (status & ExceptionFlags::UNDERFLOW ? FE_UNDERFLOW : 0) |
         (status & ExceptionFlags::INEXACT ? FE_INEXACT : 0);
}

struct X87StateDescriptor {
  uint16_t control_word;
  uint16_t unused1;
  uint16_t status_word;
  uint16_t unused2;
  // TODO: Elaborate the remaining 20 bytes as required.
  uint32_t _[5];
};

static inline uint16_t get_x87_control_word() {
  uint16_t w;
  __asm__ __volatile__("fnstcw %0" : "=m"(w)::);
  SANITIZER_MEMORY_INITIALIZED(&w, sizeof(w));
  return w;
}

static inline void write_x87_control_word(uint16_t w) {
  __asm__ __volatile__("fldcw %0" : : "m"(w) :);
}

static inline uint16_t get_x87_status_word() {
  uint16_t w;
  __asm__ __volatile__("fnstsw %0" : "=m"(w)::);
  SANITIZER_MEMORY_INITIALIZED(&w, sizeof(w));
  return w;
}

static inline void clear_x87_exceptions() {
  __asm__ __volatile__("fnclex" : : :);
}

static inline uint32_t get_mxcsr() {
  uint32_t w;
  __asm__ __volatile__("stmxcsr %0" : "=m"(w)::);
  SANITIZER_MEMORY_INITIALIZED(&w, sizeof(w));
  return w;
}

static inline void write_mxcsr(uint32_t w) {
  __asm__ __volatile__("ldmxcsr %0" : : "m"(w) :);
}

static inline void get_x87_state_descriptor(X87StateDescriptor &s) {
  __asm__ __volatile__("fnstenv %0" : "=m"(s));
  SANITIZER_MEMORY_INITIALIZED(&s, sizeof(s));
}

static inline void write_x87_state_descriptor(const X87StateDescriptor &s) {
  __asm__ __volatile__("fldenv %0" : : "m"(s) :);
}

static inline void fwait() { __asm__ __volatile__("fwait"); }

} // namespace internal

static inline int enable_except(int excepts) {
  // In the x87 control word and in MXCSR, an exception is blocked
  // if the corresponding bit is set. That is the reason for all the
  // bit-flip operations below as we need to turn the bits to zero
  // to enable them.

  uint16_t bit_mask = internal::get_status_value_for_except(excepts);

  uint16_t x87_cw = internal::get_x87_control_word();
  uint16_t old_excepts = ~x87_cw & 0x3F; // Save previously enabled exceptions.
  x87_cw &= ~bit_mask;
  internal::write_x87_control_word(x87_cw);

  // Enabling SSE exceptions via MXCSR is a nice thing to do but
  // might not be of much use practically as SSE exceptions and the x87
  // exceptions are independent of each other.
  uint32_t mxcsr = internal::get_mxcsr();
  mxcsr &= ~(bit_mask << internal::MXCSR_EXCEPTION_CONTOL_BIT_POISTION);
  internal::write_mxcsr(mxcsr);

  // Since the x87 exceptions and SSE exceptions are independent of each,
  // it doesn't make much sence to report both in the return value. Most
  // often, the standard floating point functions deal with FPU operations
  // so we will retrun only the old x87 exceptions.
  return internal::exception_status_to_macro(old_excepts);
}

static inline int disable_except(int excepts) {
  // In the x87 control word and in MXCSR, an exception is blocked
  // if the corresponding bit is set.

  uint16_t bit_mask = internal::get_status_value_for_except(excepts);

  uint16_t x87_cw = internal::get_x87_control_word();
  uint16_t old_excepts = ~x87_cw & 0x3F; // Save previously enabled exceptions.
  x87_cw |= bit_mask;
  internal::write_x87_control_word(x87_cw);

  // Just like in enable_except, it is not clear if disabling SSE exceptions
  // is required. But, we will still do it only as a "nice thing to do".
  uint32_t mxcsr = internal::get_mxcsr();
  mxcsr |= (bit_mask << internal::MXCSR_EXCEPTION_CONTOL_BIT_POISTION);
  internal::write_mxcsr(mxcsr);

  return internal::exception_status_to_macro(old_excepts);
}

static inline int get_except() {
  uint16_t x87_cw = internal::get_x87_control_word();
  uint16_t enabled_excepts = ~x87_cw & 0x3F;
  return internal::exception_status_to_macro(enabled_excepts);
}

static inline int clear_except(int excepts) {
  internal::X87StateDescriptor state;
  internal::get_x87_state_descriptor(state);
  state.status_word &= ~internal::get_status_value_for_except(excepts);
  internal::write_x87_state_descriptor(state);

  uint32_t mxcsr = internal::get_mxcsr();
  mxcsr &= ~internal::get_status_value_for_except(excepts);
  internal::write_mxcsr(mxcsr);
  return 0;
}

static inline int test_except(int excepts) {
  uint16_t status_value = internal::get_status_value_for_except(excepts);
  // Check both x87 status word and MXCSR.
  return internal::exception_status_to_macro(
      (status_value & internal::get_x87_status_word()) |
      (status_value & internal::get_mxcsr()));
}

// Sets the exception flags but does not trigger the exception handler.
static inline int set_except(int excepts) {
  uint16_t status_value = internal::get_status_value_for_except(excepts);
  internal::X87StateDescriptor state;
  internal::get_x87_state_descriptor(state);
  state.status_word |= status_value;
  internal::write_x87_state_descriptor(state);

  uint32_t mxcsr = internal::get_mxcsr();
  mxcsr |= status_value;
  internal::write_mxcsr(mxcsr);

  return 0;
}

static inline int raise_except(int excepts) {
  uint16_t status_value = internal::get_status_value_for_except(excepts);

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

  auto raise_helper = [](uint16_t singleExceptFlag) {
    internal::X87StateDescriptor state;
    internal::get_x87_state_descriptor(state);
    state.status_word |= singleExceptFlag;
    internal::write_x87_state_descriptor(state);
    internal::fwait();
  };

  if (status_value & internal::ExceptionFlags::INVALID)
    raise_helper(internal::ExceptionFlags::INVALID);
  if (status_value & internal::ExceptionFlags::DIV_BY_ZERO)
    raise_helper(internal::ExceptionFlags::DIV_BY_ZERO);
  if (status_value & internal::ExceptionFlags::OVERFLOW)
    raise_helper(internal::ExceptionFlags::OVERFLOW);
  if (status_value & internal::ExceptionFlags::UNDERFLOW)
    raise_helper(internal::ExceptionFlags::UNDERFLOW);
  if (status_value & internal::ExceptionFlags::INEXACT)
    raise_helper(internal::ExceptionFlags::INEXACT);
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

static inline int get_round() {
  uint16_t bit_value =
      (internal::get_mxcsr() >> internal::MXCSR_ROUNDING_CONTROL_BIT_POSITION) &
      0x3;
  switch (bit_value) {
  case internal::RoundingControlValue::TO_NEAREST:
    return FE_TONEAREST;
  case internal::RoundingControlValue::DOWNWARD:
    return FE_DOWNWARD;
  case internal::RoundingControlValue::UPWARD:
    return FE_UPWARD;
  case internal::RoundingControlValue::TOWARD_ZERO:
    return FE_TOWARDZERO;
  default:
    return -1; // Error value.
  }
}

static inline int set_round(int mode) {
  uint16_t bit_value;
  switch (mode) {
  case FE_TONEAREST:
    bit_value = internal::RoundingControlValue::TO_NEAREST;
    break;
  case FE_DOWNWARD:
    bit_value = internal::RoundingControlValue::DOWNWARD;
    break;
  case FE_UPWARD:
    bit_value = internal::RoundingControlValue::UPWARD;
    break;
  case FE_TOWARDZERO:
    bit_value = internal::RoundingControlValue::TOWARD_ZERO;
    break;
  default:
    return 1; // To indicate failure
  }

  uint16_t x87_value = static_cast<uint16_t>(
      bit_value << internal::X87_ROUNDING_CONTROL_BIT_POSITION);
  uint16_t x87_control = internal::get_x87_control_word();
  x87_control = static_cast<uint16_t>(
      (x87_control &
       ~(uint16_t(0x3) << internal::X87_ROUNDING_CONTROL_BIT_POSITION)) |
      x87_value);
  internal::write_x87_control_word(x87_control);

  uint32_t mxcsr_value = bit_value
                         << internal::MXCSR_ROUNDING_CONTROL_BIT_POSITION;
  uint32_t mxcsr_control = internal::get_mxcsr();
  mxcsr_control = (mxcsr_control &
                   ~(0x3 << internal::MXCSR_ROUNDING_CONTROL_BIT_POSITION)) |
                  mxcsr_value;
  internal::write_mxcsr(mxcsr_control);

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
  X87StateDescriptor x87_status;
  uint32_t mxcsr;
};
#endif // _WIN32

} // namespace internal

static_assert(
    sizeof(fenv_t) == sizeof(internal::FPState),
    "Internal floating point state does not match the public fenv_t type.");

#ifdef _WIN32
static inline int get_env(fenv_t *envp) {
  internal::FPState *state = reinterpret_cast<internal::FPState *>(envp);
  internal::X87StateDescriptor X87Status;
  internal::getX87StateDescriptor(X87Status);
  state->ControlWord = X87Status.ControlWord;
  state->StatusWord = X87Status.StatusWord;
  return 0;
}

static inline int set_env(const fenv_t *envp) {
  const internal::FPState *state =
      reinterpret_cast<const internal::FPState *>(envp);
  internal::X87StateDescriptor X87Status;
  X87Status.ControlWord = state->ControlWord;
  X87Status.StatusWord = state->StatusWord;
  internal::writeX87StateDescriptor(X87Status);
  return 0;
}
#else
static inline int get_env(fenv_t *envp) {
  internal::FPState *state = reinterpret_cast<internal::FPState *>(envp);
  internal::get_x87_state_descriptor(state->x87_status);
  state->mxcsr = internal::get_mxcsr();
  return 0;
}

static inline int set_env(const fenv_t *envp) {
  // envp contains everything including pieces like the current
  // top of FPU stack. We cannot arbitrarily change them. So, we first
  // read the current status and update only those pieces which are
  // not disruptive.
  internal::X87StateDescriptor x87_status;
  internal::get_x87_state_descriptor(x87_status);

  if (envp == FE_DFL_ENV) {
    // Reset the exception flags in the status word.
    x87_status.status_word &= ~uint16_t(0x3F);
    // Reset other non-sensitive parts of the status word.
    for (int i = 0; i < 5; i++)
      x87_status._[i] = 0;
    // In the control word, we do the following:
    // 1. Mask all exceptions
    // 2. Set rounding mode to round-to-nearest
    // 3. Set the internal precision to double extended precision.
    x87_status.control_word |= uint16_t(0x3F);         // Mask all exceptions.
    x87_status.control_word &= ~(uint16_t(0x3) << 10); // Round to nearest.
    x87_status.control_word |= (uint16_t(0x3) << 8);   // Extended precision.
    internal::write_x87_state_descriptor(x87_status);

    // We take the exact same approach MXCSR register as well.
    // MXCSR has two additional fields, "flush-to-zero" and
    // "denormals-are-zero". We reset those bits. Also, MXCSR does not
    // have a field which controls the precision of internal operations.
    uint32_t mxcsr = internal::get_mxcsr();
    mxcsr &= ~uint16_t(0x3F);        // Clear exception flags.
    mxcsr &= ~(uint16_t(0x1) << 6);  // Reset denormals-are-zero
    mxcsr |= (uint16_t(0x3F) << 7);  // Mask exceptions
    mxcsr &= ~(uint16_t(0x3) << 13); // Round to nearest.
    mxcsr &= ~(uint16_t(0x1) << 15); // Reset flush-to-zero
    internal::write_mxcsr(mxcsr);

    return 0;
  }

  const internal::FPState *fpstate =
      reinterpret_cast<const internal::FPState *>(envp);

  // Copy the exception status flags from envp.
  x87_status.status_word &= ~uint16_t(0x3F);
  x87_status.status_word |= (fpstate->x87_status.status_word & 0x3F);
  // Copy other non-sensitive parts of the status word.
  for (int i = 0; i < 5; i++)
    x87_status._[i] = fpstate->x87_status._[i];
  // We can set the x87 control word as is as there no sensitive bits.
  x87_status.control_word = fpstate->x87_status.control_word;
  internal::write_x87_state_descriptor(x87_status);

  // We can write the MXCSR state as is as there are no sensitive bits.
  internal::write_mxcsr(fpstate->mxcsr);
  return 0;
}
#endif

} // namespace fputil
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_FPUTIL_X86_64_FENVIMPL_H
