//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// DWARF register numbers for x86_64 (AMD64, Intel EM64T, or Intel 64).
///
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_ARCH_LLDB_X86_64_DWARF_REGISTER_ENUMS_H
#define LLDB_REGISTER_ARCH_LLDB_X86_64_DWARF_REGISTER_ENUMS_H

#include <cstdint>

namespace lldb_private::registers::x86_64::dwarf {

// DWARF register numbers (eRegisterKindDWARF)
enum : uint32_t {
  // General Purpose Registers.
  rax = 0,
  rdx,
  rcx,
  rbx,
  rsi,
  rdi,
  rbp,
  rsp,

  // Extended GP Registers.
  r8 = 8,
  r9,
  r10,
  r11,
  r12,
  r13,
  r14,
  r15,
  // Return Address (RA) mapped to RIP
  rip = 16,

  // SSE Vector Registers.
  xmm0 = 17,
  xmm1,
  xmm2,
  xmm3,
  xmm4,
  xmm5,
  xmm6,
  xmm7,
  xmm8,
  xmm9,
  xmm10,
  xmm11,
  xmm12,
  xmm13,
  xmm14,
  xmm15,

  // Floating Point Stack Registers.
  st0 = 33,
  st1,
  st2,
  st3,
  st4,
  st5,
  st6,
  st7,

  // MMX Registers.
  mm0 = 41,
  mm1,
  mm2,
  mm3,
  mm4,
  mm5,
  mm6,
  mm7,

  // General Purpose Registers.
  rflags = 49,

  // Segment Registers.
  es = 50,
  cs,
  ss,
  ds,
  fs,
  gs,

  // Segment Base Registers.
  fs_base = 58,
  gs_base = 59,

  // SSE Control and Status Registers.
  mxcsr = 64, // Media Control and Status

  // Floating Point Control and Status Registers.
  fctrl,      // x87 control word
  fstat,      // x87 status word

  // AVX Vector Registers.
  ymm0h = 67,
  ymm1h,
  ymm2h,
  ymm3h,
  ymm4h,
  ymm5h,
  ymm6h,
  ymm7h,
  ymm8h,
  ymm9h,
  ymm10h,
  ymm11h,
  ymm12h,
  ymm13h,
  ymm14h,
  ymm15h,

  // MPX Bound Registers.
  bnd0 = 126,
  bnd1,
  bnd2,
  bnd3,

  // AVX2 Vector Mask Registers.
  // dwarf_k0_x86_64 = 118,
  // dwarf_k1_x86_64,
  // dwarf_k2_x86_64,
  // dwarf_k3_x86_64,
  // dwarf_k4_x86_64,
  // dwarf_k5_x86_64,
  // dwarf_k6_x86_64,
  // dwarf_k7_x86_64,
};

} // namespace lldb_private::registers::x86_64::dwarf

#endif // LLDB_REGISTER_ARCH_LLDB_X86_64_DWARF_REGISTER_ENUMS_H
