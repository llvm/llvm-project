//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// DWARF register numbers for arm.
///
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_ARCH_LLDB_ARM_DWARF_REGISTER_ENUMS_H
#define LLDB_REGISTER_ARCH_LLDB_ARM_DWARF_REGISTER_ENUMS_H

#include <cstdint>

namespace lldb_private::registers::arm::dwarf {

// DWARF register numbers (eRegisterKindDWARF)
enum : uint32_t {
  // General Purpose Registers.
  r0 = 0,
  r1,
  r2,
  r3,
  r4,
  r5,
  r6,
  r7,
  r8,
  r9,
  r10,
  r11,
  r12,
  sp,
  lr,
  pc,

  // Status Registers.
  cpsr,

  // Floating Point and Vector Registers.
  s0 = 64,
  s1,
  s2,
  s3,
  s4,
  s5,
  s6,
  s7,
  s8,
  s9,
  s10,
  s11,
  s12,
  s13,
  s14,
  s15,
  s16,
  s17,
  s18,
  s19,
  s20,
  s21,
  s22,
  s23,
  s24,
  s25,
  s26,
  s27,
  s28,
  s29,
  s30,
  s31,

  // Floating Point Registers.
  // FPA Registers 0-7
  f0 = 96,
  f1,
  f2,
  f3,
  f4,
  f5,
  f6,
  f7,

  // Intel wireless MMX general purpose registers 0 - 7
  wCGR0 = 104,
  wCGR1,
  wCGR2,
  wCGR3,
  wCGR4,
  wCGR5,
  wCGR6,
  wCGR7,

  // XScale accumulator register 0 - 7 (they do overlap with wCGR0 - wCGR7)
  ACC0 = 104,
  ACC1,
  ACC2,
  ACC3,
  ACC4,
  ACC5,
  ACC6,
  ACC7,

  // Intel wireless MMX data registers 0 - 15
  wR0 = 112,
  wR1,
  wR2,
  wR3,
  wR4,
  wR5,
  wR6,
  wR7,
  wR8,
  wR9,
  wR10,
  wR11,
  wR12,
  wR13,
  wR14,
  wR15,

  // Saved Program Status Registers.
  spsr = 128,
  spsr_fiq,
  spsr_irq,
  spsr_abt,
  spsr_und,
  spsr_svc,

  // Banked General Purpose Registers.
  r8_usr = 144,
  r9_usr,
  r10_usr,
  r11_usr,
  r12_usr,
  r13_usr,
  r14_usr,
  r8_fiq,
  r9_fiq,
  r10_fiq,
  r11_fiq,
  r12_fiq,
  r13_fiq,
  r14_fiq,
  r13_irq,
  r14_irq,
  r13_abt,
  r14_abt,
  r13_und,
  r14_und,
  r13_svc,
  r14_svc,

  // Intel wireless MMX control register in co-processor 0 - 7
  wC0 = 192,
  wC1,
  wC2,
  wC3,
  wC4,
  wC5,
  wC6,
  wC7,

  // Floating Point and Vector Registers.
  // VFP-v3/Neon
  d0 = 256,
  d1,
  d2,
  d3,
  d4,
  d5,
  d6,
  d7,
  d8,
  d9,
  d10,
  d11,
  d12,
  d13,
  d14,
  d15,
  d16,
  d17,
  d18,
  d19,
  d20,
  d21,
  d22,
  d23,
  d24,
  d25,
  d26,
  d27,
  d28,
  d29,
  d30,
  d31,

  // Neon quadword registers.
  q0 = 288,
  q1,
  q2,
  q3,
  q4,
  q5,
  q6,
  q7,
  q8,
  q9,
  q10,
  q11,
  q12,
  q13,
  q14,
  q15
};

} // namespace lldb_private::registers::arm::dwarf

#endif // LLDB_REGISTER_ARCH_LLDB_ARM_DWARF_REGISTER_ENUMS_H
