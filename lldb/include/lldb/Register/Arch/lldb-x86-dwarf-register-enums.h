//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// DWARF register numbers for Intel x86 (IA-32).
///
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_ARCH_LLDB_X86_DWARF_REGISTER_ENUMS_H
#define LLDB_REGISTER_ARCH_LLDB_X86_DWARF_REGISTER_ENUMS_H

#include <cstdint>

namespace lldb_private::registers::x86::dwarf {

// DWARF register numbers (eRegisterKindDWARF)
enum : uint32_t {
  // General Purpose Registers.
  eax = 0,
  ecx,
  edx,
  ebx,
  esp,
  ebp,
  esi,
  edi,
  eip,
  eflags,

  // Floating Point Stack Registers.
  st0 = 11,
  st1,
  st2,
  st3,
  st4,
  st5,
  st6,
  st7,

  // SSE Vector Registers.
  xmm0 = 21,
  xmm1,
  xmm2,
  xmm3,
  xmm4,
  xmm5,
  xmm6,
  xmm7,

  // MMX Registers.
  mm0 = 29,
  mm1,
  mm2,
  mm3,
  mm4,
  mm5,
  mm6,
  mm7,

  // Floating Point Control and Status Registers.
  fctrl = 37, // x87 control word
  fstat = 38, // x87 status word

  // SSE Control and Status Registers.
  mxcsr = 39,

  // Segment Registers.
  es = 40,
  cs = 41,
  ss = 42,
  ds = 43,
  fs = 44,
  gs = 45,

  // MPX Bound Registers.
  // I believe the ymm registers use the dwarf_xmm%_i386 register numbers and
  bnd0 = 101,
  bnd1,
  bnd2,
  bnd3,
};

} // namespace lldb_private::registers::x86::dwarf

#endif // LLDB_REGISTER_ARCH_LLDB_X86_DWARF_REGISTER_ENUMS_H
