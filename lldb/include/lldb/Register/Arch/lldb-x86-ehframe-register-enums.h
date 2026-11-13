//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// EH-frame register numbers for x86.
///
/// x86 uses different DWARF and EH-frame register numbers.
/// For example, ST0 is 11 in DWARF and 12 in EH-frame.
///
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_ARCH_LLDB_X86_EHFRAME_REGISTER_ENUMS_H
#define LLDB_REGISTER_ARCH_LLDB_X86_EHFRAME_REGISTER_ENUMS_H

#include "lldb/Register/Arch/lldb-x86-dwarf-register-enums.h"
#include <cstdint>

namespace lldb_private::registers::x86::ehframe {

// EH-frame register numbers (eRegisterKindEHFrame)
enum : uint32_t {
  // General Purpose Registers.
  eax = 0,
  ecx,
  edx,
  ebx,

  // on Darwin esp & ebp are reversed in the eh_frame section for i386 (versus
  // dwarf's reg numbering).
  // To be specific:
  //    i386+darwin eh_frame:        4 is ebp, 5 is esp
  //    i386+everyone else eh_frame: 4 is esp, 5 is ebp
  //    i386 dwarf:                  4 is esp, 5 is ebp
  // lldb will get the darwin-specific eh_frame reg numberings from debugserver,
  // or the ABI, so we
  // only encode the generally correct 4 == esp, 5 == ebp numbers in this
  // generic header.

  esp,
  ebp,
  esi,
  edi,
  eip,
  eflags,

  // Floating Point Stack Registers.
  st0 = 12,
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
  fctrl = dwarf::fctrl,
  fstat = dwarf::fstat,

  // SSE Control and Status Registers.
  mxcsr = dwarf::mxcsr,

  // Segment Registers.
  es = dwarf::es,
  cs = dwarf::cs,
  ss = dwarf::ss,
  ds = dwarf::ds,
  fs = dwarf::fs,
  gs = dwarf::gs,

  // MPX Bound Registers.
  bnd0 = dwarf::bnd0,
  bnd1 = dwarf::bnd1,
  bnd2 = dwarf::bnd2,
  bnd3 = dwarf::bnd3,
};

} // namespace lldb_private::registers::x86::ehframe

#endif // LLDB_REGISTER_ARCH_LLDB_X86_EHFRAME_REGISTER_ENUMS_H
