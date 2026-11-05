//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// LLDB register IDs for x86_64.
///
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_ARCH_LLDB_X86_64_REGISTER_ENUMS_H
#define LLDB_REGISTER_ARCH_LLDB_X86_64_REGISTER_ENUMS_H

#include <cstdint>

namespace lldb_private::registers::x86_64::id {

// LLDB register codes (e.g. RegisterKind == eRegisterKindLLDB)
enum : uint32_t {
  // General Purpose Registers.
  k_first_gpr,
  rax = k_first_gpr,
  rbx,
  rcx,
  rdx,
  rdi,
  rsi,
  rbp,
  rsp,
  r8,
  r9,
  r10,
  r11,
  r12,
  r13,
  r14,
  r15,
  rip,
  rflags,

  // Segment Registers.
  cs,
  fs,
  gs,
  ss,
  ds,
  es,

  // Subregister Aliases.
  k_first_alias,
  eax = k_first_alias,
  ebx,
  ecx,
  edx,
  edi,
  esi,
  ebp,
  esp,
  r8d,  // Low 32 bits of r8
  r9d,  // Low 32 bits of r9
  r10d, // Low 32 bits of r10
  r11d, // Low 32 bits of r11
  r12d, // Low 32 bits of r12
  r13d, // Low 32 bits of r13
  r14d, // Low 32 bits of r14
  r15d, // Low 32 bits of r15
  ax,
  bx,
  cx,
  dx,
  di,
  si,
  bp,
  sp,
  r8w,  // Low 16 bits of r8
  r9w,  // Low 16 bits of r9
  r10w, // Low 16 bits of r10
  r11w, // Low 16 bits of r11
  r12w, // Low 16 bits of r12
  r13w, // Low 16 bits of r13
  r14w, // Low 16 bits of r14
  r15w, // Low 16 bits of r15
  ah,
  bh,
  ch,
  dh,
  al,
  bl,
  cl,
  dl,
  dil,
  sil,
  bpl,
  spl,
  r8l,  // Low 8 bits of r8
  r9l,  // Low 8 bits of r9
  r10l, // Low 8 bits of r10
  r11l, // Low 8 bits of r11
  r12l, // Low 8 bits of r12
  r13l, // Low 8 bits of r13
  r14l, // Low 8 bits of r14
  r15l, // Low 8 bits of r15
  k_last_alias = r15l,

  k_last_gpr = k_last_alias,

  // Floating Point Registers.
  k_first_fpr,

  // Floating Point Control and Status Registers.
  fctrl = k_first_fpr,
  fstat,
  ftag,
  fop,
  fiseg,
  fioff,
  fip,
  foseg,
  fooff,
  fdp,

  // SSE Control and Status Registers.
  mxcsr,
  mxcsrmask,

  // Floating Point Stack Registers.
  st0,
  st1,
  st2,
  st3,
  st4,
  st5,
  st6,
  st7,

  // MMX Registers.
  mm0,
  mm1,
  mm2,
  mm3,
  mm4,
  mm5,
  mm6,
  mm7,

  // SSE Vector Registers.
  xmm0,
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
  k_last_fpr = xmm15,

  // AVX Vector Registers.
  k_first_avx,
  ymm0 = k_first_avx,
  ymm1,
  ymm2,
  ymm3,
  ymm4,
  ymm5,
  ymm6,
  ymm7,
  ymm8,
  ymm9,
  ymm10,
  ymm11,
  ymm12,
  ymm13,
  ymm14,
  ymm15,
  k_last_avx = ymm15,

  // MPX Bound Registers.
  k_first_mpxr,
  bnd0 = k_first_mpxr,
  bnd1,
  bnd2,
  bnd3,
  k_last_mpxr = bnd3,

  // MPX Control Registers.
  k_first_mpxc,
  bndcfgu = k_first_mpxc,
  bndstatus,
  k_last_mpxc = bndstatus,

  // Debug Registers.
  k_first_dbr,
  dr0 = k_first_dbr,
  dr1,
  dr2,
  dr3,
  dr4,
  dr5,
  dr6,
  dr7,
  k_last_dbr = dr7,

  // Register Counts.
  k_num_registers,
  k_num_gpr_registers = k_last_gpr - k_first_gpr + 1,
  k_num_fpr_registers = k_last_fpr - k_first_fpr + 1,
  k_num_avx_registers = k_last_avx - k_first_avx + 1,
  k_num_mpx_registers = k_last_mpxc - k_first_mpxr + 1,
  k_num_user_registers = k_num_gpr_registers + k_num_fpr_registers +
      k_num_avx_registers + k_num_mpx_registers,
  k_num_dbr_registers = k_last_dbr - k_first_dbr + 1,
};

} // namespace lldb_private::registers::x86_64::id
namespace lldb_private::registers::x86_64::id::with_base {

enum : uint32_t {
  // General Purpose Registers.
  k_first_gpr,
  rax = k_first_gpr,
  rbx,
  rcx,
  rdx,
  rdi,
  rsi,
  rbp,
  rsp,
  r8,
  r9,
  r10,
  r11,
  r12,
  r13,
  r14,
  r15,
  rip,
  rflags,

  // Segment Registers.
  cs,
  fs,
  gs,
  ss,

  // Segment Base Registers.
  fs_base,
  gs_base,

  // Segment Registers.
  ds,
  es,

  // Subregister Aliases.
  k_first_alias,
  eax = k_first_alias,
  ebx,
  ecx,
  edx,
  edi,
  esi,
  ebp,
  esp,
  r8d,  // Low 32 bits of r8
  r9d,  // Low 32 bits of r9
  r10d, // Low 32 bits of r10
  r11d, // Low 32 bits of r11
  r12d, // Low 32 bits of r12
  r13d, // Low 32 bits of r13
  r14d, // Low 32 bits of r14
  r15d, // Low 32 bits of r15
  ax,
  bx,
  cx,
  dx,
  di,
  si,
  bp,
  sp,
  r8w,  // Low 16 bits of r8
  r9w,  // Low 16 bits of r9
  r10w, // Low 16 bits of r10
  r11w, // Low 16 bits of r11
  r12w, // Low 16 bits of r12
  r13w, // Low 16 bits of r13
  r14w, // Low 16 bits of r14
  r15w, // Low 16 bits of r15
  ah,
  bh,
  ch,
  dh,
  al,
  bl,
  cl,
  dl,
  dil,
  sil,
  bpl,
  spl,
  r8l,  // Low 8 bits of r8
  r9l,  // Low 8 bits of r9
  r10l, // Low 8 bits of r10
  r11l, // Low 8 bits of r11
  r12l, // Low 8 bits of r12
  r13l, // Low 8 bits of r13
  r14l, // Low 8 bits of r14
  r15l, // Low 8 bits of r15
  k_last_alias = r15l,

  k_last_gpr = k_last_alias,

  // Floating Point Registers.
  k_first_fpr,

  // Floating Point Control and Status Registers.
  fctrl = k_first_fpr,
  fstat,
  ftag,
  fop,
  fiseg,
  fioff,
  fip,
  foseg,
  fooff,
  fdp,

  // SSE Control and Status Registers.
  mxcsr,
  mxcsrmask,

  // Floating Point Stack Registers.
  st0,
  st1,
  st2,
  st3,
  st4,
  st5,
  st6,
  st7,

  // MMX Registers.
  mm0,
  mm1,
  mm2,
  mm3,
  mm4,
  mm5,
  mm6,
  mm7,

  // SSE Vector Registers.
  xmm0,
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
  k_last_fpr = xmm15,

  // AVX Vector Registers.
  k_first_avx,
  ymm0 = k_first_avx,
  ymm1,
  ymm2,
  ymm3,
  ymm4,
  ymm5,
  ymm6,
  ymm7,
  ymm8,
  ymm9,
  ymm10,
  ymm11,
  ymm12,
  ymm13,
  ymm14,
  ymm15,
  k_last_avx = ymm15,

  // MPX Bound Registers.
  k_first_mpxr,
  bnd0 = k_first_mpxr,
  bnd1,
  bnd2,
  bnd3,
  k_last_mpxr = bnd3,

  // MPX Control Registers.
  k_first_mpxc,
  bndcfgu = k_first_mpxc,
  bndstatus,
  k_last_mpxc = bndstatus,

  // Debug Registers.
  k_first_dbr,
  dr0 = k_first_dbr,
  dr1,
  dr2,
  dr3,
  dr4,
  dr5,
  dr6,
  dr7,
  k_last_dbr = dr7,

  // Register Counts.
  k_num_registers,
  k_num_gpr_registers = k_last_gpr - k_first_gpr + 1,
  k_num_fpr_registers = k_last_fpr - k_first_fpr + 1,
  k_num_avx_registers = k_last_avx - k_first_avx + 1,
  k_num_mpx_registers = k_last_mpxc - k_first_mpxr + 1,
  k_num_user_registers = k_num_gpr_registers + k_num_fpr_registers +
      k_num_avx_registers + k_num_mpx_registers,
  k_num_dbr_registers = k_last_dbr - k_first_dbr + 1,
};

} // namespace lldb_private::registers::x86_64::id::with_base

#endif // LLDB_REGISTER_ARCH_LLDB_X86_64_REGISTER_ENUMS_H
