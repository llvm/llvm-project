//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// DWARF register numbers for mips32.
///
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_ARCH_LLDB_MIPS32_DWARF_REGISTER_ENUMS_H
#define LLDB_REGISTER_ARCH_LLDB_MIPS32_DWARF_REGISTER_ENUMS_H

#include <cstdint>

namespace lldb_private::registers::mips32::dwarf {

// DWARF register numbers (eRegisterKindDWARF)
enum : uint32_t {
  // General Purpose Registers.
  zero = 0,
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
  r13,
  r14,
  r15,
  r16,
  r17,
  r18,
  r19,
  r20,
  r21,
  r22,
  r23,
  r24,
  r25,
  r26,
  r27,
  gp,
  sp,
  r30,
  ra,

  // Control and Status Registers.
  sr,
  lo,
  hi,
  bad,
  cause,

  // General Purpose Registers.
  pc,

  // Floating Point Registers.
  f0,
  f1,
  f2,
  f3,
  f4,
  f5,
  f6,
  f7,
  f8,
  f9,
  f10,
  f11,
  f12,
  f13,
  f14,
  f15,
  f16,
  f17,
  f18,
  f19,
  f20,
  f21,
  f22,
  f23,
  f24,
  f25,
  f26,
  f27,
  f28,
  f29,
  f30,
  f31,

  // Floating Point Control and Status Registers.
  fcsr,
  fir,

  // MSA Vector Registers.
  w0,
  w1,
  w2,
  w3,
  w4,
  w5,
  w6,
  w7,
  w8,
  w9,
  w10,
  w11,
  w12,
  w13,
  w14,
  w15,
  w16,
  w17,
  w18,
  w19,
  w20,
  w21,
  w22,
  w23,
  w24,
  w25,
  w26,
  w27,
  w28,
  w29,
  w30,
  w31,
  mcsr,
  mir,

  // Control and Status Registers.
  config5,
  ic,
  dummy
};

} // namespace lldb_private::registers::mips32::dwarf

#endif // LLDB_REGISTER_ARCH_LLDB_MIPS32_DWARF_REGISTER_ENUMS_H
