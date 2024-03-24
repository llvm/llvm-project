//===-- amdgpu floating point env manipulation functions --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FPUTIL_AMDGPU_FENVIMPL_H
#define LLVM_LIBC_SRC___SUPPORT_FPUTIL_AMDGPU_FENVIMPL_H

#include "src/__support/GPU/utils.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/properties/architectures.h"

#if !defined(LIBC_TARGET_ARCH_IS_AMDGPU)
#error "Invalid include"
#endif

#include <fenv.h>
#include <stdint.h>

namespace LIBC_NAMESPACE {
namespace fputil {

namespace internal {

// Gets the immediate argument to access the AMDGPU hardware register. The
// register access is encoded in a 16-bit immediate value according to the
// following layout.
//
// ┌──────────────┬──────────────┬───────────────┐
// │  SIZE[15:11] │ OFFSET[10:6] │    ID[5:0]    │
// └──────────────┴──────────────┴───────────────┘
//
// This will read the size number of bits starting at the offset bit from the
// corresponding hardware register ID.
LIBC_INLINE constexpr uint16_t get_register(uint32_t id, uint32_t offset,
                                            uint32_t size) {
  return static_cast<uint16_t>(size << 11 | offset << 6 | id);
}

// Integral identifiers for the relevant hardware registers.
enum Register : uint16_t {
  // The mode register controls the floating point behaviour of the device. It
  // can be read or written to by the kernel during runtime It is laid out as a
  // bit field with the following offsets and sizes listed for the relevant
  // entries.
  //
  // ┌─────┬─────────────┬─────┬─────────┬──────────┬─────────────┬────────────┐
  // │ ... │ EXCP[20:12] │ ... │ IEEE[9] │ CLAMP[8] │ DENORM[7:4] │ ROUND[3:0] │
  // └─────┴─────────────┴─────┴─────────┴──────────┴─────────────┴────────────┘
  //
  // The rounding mode and denormal modes both control f64/f16 and f32 precision
  // operations separately with two bits. The accepted values for the rounding
  // mode are nearest, upward, downward, and toward given 0, 1, 2, and 3
  // respectively.
  //
  // The CLAMP bit indicates that DirectX 10 handling of NaNs is enabled in the
  // vector ALU. When set this will clamp NaN values to zero and pass them
  // otherwise. A hardware bug causes this bit to prevent floating exceptions
  // from being recorded if this bit is set on all generations before GFX12.
  //
  // The IEEE bit controls whether or not floating point operations supporting
  // exception gathering are IEEE 754-2008 compliant.
  //
  // The EXCP field indicates which exceptions will cause the instruction to
  // take a trap if traps are enabled, see the status register. The bit layout
  // is identical to that in the trap status register. We are only concerned
  // with the first six bits and ignore the other three.
  HW_REG_MODE = 1,
  HW_REG_MODE_ROUND = get_register(HW_REG_MODE, 0, 4),
  HW_REG_MODE_CLAMP = get_register(HW_REG_MODE, 8, 1),
  HW_REG_MODE_EXCP = get_register(HW_REG_MODE, 12, 6),

  // The status register is a read-only register that contains information about
  // how the kernel was launched. The sixth bit TRAP_EN[6] indicates whether or
  // not traps are enabled for this kernel. If this bit is set along with the
  // corresponding bit in the mode register then a trap will be taken.
  HW_REG_STATUS = 2,
  HW_REG_STATUS_TRAP_EN = get_register(HW_REG_STATUS, 6, 1),

  // The trap status register contains information about the status of the
  // exceptions. These bits are accumulated regarless of trap handling statuss
  // and are sticky until cleared.
  //
  // 5         4           3          2                1          0
  // ┌─────────┬───────────┬──────────┬────────────────┬──────────┬─────────┐
  // │ Inexact │ Underflow │ Overflow │ Divide by zero │ Denormal │ Invalid │
  // └─────────┴───────────┴──────────┴────────────────┴──────────┴─────────┘
  //
  // These exceptions indicate that at least one lane in the current wavefront
  // signalled an floating point exception. There is no way to increase the
  // granularity.
  HW_REG_TRAPSTS = 3,
  HW_REG_TRAPSTS_EXCP = get_register(HW_REG_TRAPSTS, 0, 6),
};

// The six bits used to encode the standard floating point exceptions in the
// trap status register.
enum ExceptionFlags : uint32_t {
  EXCP_INVALID_F = 0x1,
  EXCP_DENORMAL_F = 0x2,
  EXCP_DIV_BY_ZERO_F = 0x4,
  EXCP_OVERFLOW_F = 0x8,
  EXCP_UNDERFLOW_F = 0x10,
  EXCP_INEXACT_F = 0x20,
};

// The two bit encoded rounding modes used in the mode register.
enum RoundingFlags : uint32_t {
  ROUND_TO_NEAREST = 0x0,
  ROUND_UPWARD = 0x1,
  ROUND_DOWNWARD = 0x2,
  ROUND_TOWARD_ZERO = 0x3,
};

// Exception flags are individual bits in the corresponding hardware register.
// This converts between the exported C standard values and the hardware values.
LIBC_INLINE uint32_t get_status_value_for_except(uint32_t excepts) {
  return (excepts & FE_INVALID ? EXCP_INVALID_F : 0) |
#ifdef __FE_DENORM
         (excepts & __FE_DENORM ? EXCP_DENORMAL_F : 0) |
#endif // __FE_DENORM
         (excepts & FE_DIVBYZERO ? EXCP_DIV_BY_ZERO_F : 0) |
         (excepts & FE_OVERFLOW ? EXCP_OVERFLOW_F : 0) |
         (excepts & FE_UNDERFLOW ? EXCP_UNDERFLOW_F : 0) |
         (excepts & FE_INEXACT ? EXCP_INEXACT_F : 0);
}

LIBC_INLINE uint32_t get_except_value_for_status(uint32_t status) {
  return (status & EXCP_INVALID_F ? FE_INVALID : 0) |
#ifdef __FE_DENORM
         (status & EXCP_DENORMAL_F ? __FE_DENORM : 0) |
#endif // __FE_DENORM
         (status & EXCP_DIV_BY_ZERO_F ? FE_DIVBYZERO : 0) |
         (status & EXCP_OVERFLOW_F ? FE_OVERFLOW : 0) |
         (status & EXCP_UNDERFLOW_F ? FE_UNDERFLOW : 0) |
         (status & EXCP_INEXACT_F ? FE_INEXACT : 0);
}

// FIXME: These require the 'noinline' attribute to pessimistically flush the
//        state. Otherwise, reading from the register may return stale results.

// Access the six bits in the trap status register for the floating point
// exceptions.
[[gnu::noinline]] LIBC_INLINE void set_trap_status(uint32_t status) {
  uint32_t val = gpu::broadcast_value(gpu::get_lane_mask(), status);
  __builtin_amdgcn_s_setreg(HW_REG_TRAPSTS_EXCP, val);
}

[[gnu::noinline]] LIBC_INLINE uint32_t get_trap_status() {
  return __builtin_amdgcn_s_getreg(HW_REG_TRAPSTS_EXCP);
}

// Access the six bits in the mode register that control which exceptions will
// result in a trap being taken. Uses the same flags as the status register.
[[gnu::noinline]] LIBC_INLINE void set_enabled_trap(uint32_t flags) {
  uint32_t val = gpu::broadcast_value(gpu::get_lane_mask(), flags);
  __builtin_amdgcn_s_setreg(HW_REG_MODE_EXCP, val);
}

[[gnu::noinline]] LIBC_INLINE uint32_t get_enabled_trap() {
  return __builtin_amdgcn_s_getreg(HW_REG_MODE_EXCP);
}

// Access the four bits in the mode register's ROUND[3:0] field. The hardware
// supports setting the f64/f16 and f32 precision rounding modes separately but
// we will assume that these always match.
[[gnu::noinline]] LIBC_INLINE void set_rounding_mode(uint32_t flags) {
  uint32_t val = gpu::broadcast_value(gpu::get_lane_mask(), flags);
  __builtin_amdgcn_s_setreg(HW_REG_MODE_ROUND, val << 2 | val);
}

[[gnu::noinline]] LIBC_INLINE uint32_t get_rounding_mode() {
  return __builtin_amdgcn_s_getreg(HW_REG_MODE_ROUND) & 0x3;
}

// NOTE: On architectures before GFX12 the DX10_CLAMP bit supresses all floating
//       point exceptions. In order to get them to be presented we need to
//       manually set if off.
[[gnu::noinline]] LIBC_INLINE void set_clamp_low() {
  __builtin_amdgcn_s_setreg(HW_REG_MODE_CLAMP, 0);
}

[[gnu::noinline]] LIBC_INLINE void set_clamp_high() {
  __builtin_amdgcn_s_setreg(HW_REG_MODE_CLAMP, 1);
}

} // namespace internal

LIBC_INLINE int clear_except(int excepts) {
  uint32_t status = internal::get_status_value_for_except(excepts);
  uint32_t invert = ~status & 0x3f;
  uint32_t active = internal::get_trap_status();
  internal::set_trap_status(active & invert);
  return 0;
}

LIBC_INLINE int test_except(int excepts) {
  uint32_t status = internal::get_status_value_for_except(excepts);
  uint32_t active = internal::get_trap_status();
  return internal::get_except_value_for_status(active) & status;
}

LIBC_INLINE int get_except() { return internal::get_trap_status(); }

LIBC_INLINE int set_except(int excepts) {
  internal::set_trap_status(internal::get_status_value_for_except(excepts));
  return 0;
}

LIBC_INLINE int enable_except(int excepts) {
  uint32_t status = internal::get_status_value_for_except(excepts);
  uint32_t active = internal::get_trap_status();
  internal::set_enabled_trap(status);
  return internal::get_except_value_for_status(active);
}

LIBC_INLINE int disable_except(int excepts) {
  uint32_t status = internal::get_status_value_for_except(excepts);
  uint32_t invert = ~status & 0x3f;
  uint32_t active = internal::get_enabled_trap();
  internal::set_enabled_trap(active & invert);
  return active;
}

LIBC_INLINE int raise_except(int excepts) {
  uint32_t status = internal::get_status_value_for_except(excepts);
  enable_except(status);
  internal::set_trap_status(status);
  return 0;
}

LIBC_INLINE int get_round() {
  switch (internal::get_rounding_mode()) {
  case internal::ROUND_TO_NEAREST:
    return FE_TONEAREST;
  case internal::ROUND_UPWARD:
    return FE_UPWARD;
  case internal::ROUND_DOWNWARD:
    return FE_DOWNWARD;
  case internal::ROUND_TOWARD_ZERO:
    return FE_TOWARDZERO;
  }
  __builtin_unreachable();
}

LIBC_INLINE int set_round(int rounding_mode) {
  switch (rounding_mode) {
  case FE_TONEAREST:
    internal::set_rounding_mode(internal::ROUND_TO_NEAREST);
    break;
  case FE_UPWARD:
    internal::set_rounding_mode(internal::ROUND_UPWARD);
    break;
  case FE_DOWNWARD:
    internal::set_rounding_mode(internal::ROUND_DOWNWARD);
    break;
  case FE_TOWARDZERO:
    internal::set_rounding_mode(internal::ROUND_TOWARD_ZERO);
    break;
  default:
    return 1;
  }
  return 0;
}

// The fenv_t struct for the AMD GPU is simply a 32-bit integer field of the
// current state. We combine the four bits for the rounding mode with the six
// bits for the exception state and the six bits for the enabled exceptions.
//
// ┌────────────────────────────┬─────────────────┬─────────────┬─────────────┐
// │       UNUSED[31:16]        │ ENABLED[15:10]  │ STATUS[9:4] │  ROUND[3:0] │
// └────────────────────────────┴─────────────────┴─────────────┴─────────────┘
//
// The top sixteen bits are currently unused and should be zero.
LIBC_INLINE int get_env(fenv_t *env) {
  if (!env)
    return 1;

  uint32_t rounding = internal::get_rounding_mode();
  uint32_t status = internal::get_trap_status();
  uint32_t enabled = internal::get_enabled_trap();
  env->__fpc = enabled << 10 | status << 4 | rounding;
  return 0;
}

LIBC_INLINE int set_env(const fenv_t *env) {
  if (!env)
    return 1;

  internal::set_rounding_mode(env->__fpc & 0xf);
  internal::set_trap_status((env->__fpc >> 4) & 0x3f);
  internal::set_enabled_trap((env->__fpc >> 10) & 0x3f);
  return 0;
}

} // namespace fputil
} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC___SUPPORT_FPUTIL_AMDGPU_FENVIMPL_H
