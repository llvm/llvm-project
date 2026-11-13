//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// LLDB register IDs for powerpc.
///
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_ARCH_LLDB_POWERPC_REGISTER_ENUMS_H
#define LLDB_REGISTER_ARCH_LLDB_POWERPC_REGISTER_ENUMS_H

#include <cstdint>

namespace lldb_private::registers::powerpc::id {

// LLDB register codes (e.g. RegisterKind == eRegisterKindLLDB)
enum : uint32_t {
  // General Purpose Registers.
  k_first_gpr,
  gpr_r0 = k_first_gpr,
  gpr_r1,
  gpr_r2,
  gpr_r3,
  gpr_r4,
  gpr_r5,
  gpr_r6,
  gpr_r7,
  gpr_r8,
  gpr_r9,
  gpr_r10,
  gpr_r11,
  gpr_r12,
  gpr_r13,
  gpr_r14,
  gpr_r15,
  gpr_r16,
  gpr_r17,
  gpr_r18,
  gpr_r19,
  gpr_r20,
  gpr_r21,
  gpr_r22,
  gpr_r23,
  gpr_r24,
  gpr_r25,
  gpr_r26,
  gpr_r27,
  gpr_r28,
  gpr_r29,
  gpr_r30,
  gpr_r31,
  gpr_lr,
  gpr_cr,
  gpr_xer,
  gpr_ctr,
  gpr_pc,
  k_last_gpr = gpr_pc,

  // Floating Point Registers.
  k_first_fpr,
  fpr_f0 = k_first_fpr,
  fpr_f1,
  fpr_f2,
  fpr_f3,
  fpr_f4,
  fpr_f5,
  fpr_f6,
  fpr_f7,
  fpr_f8,
  fpr_f9,
  fpr_f10,
  fpr_f11,
  fpr_f12,
  fpr_f13,
  fpr_f14,
  fpr_f15,
  fpr_f16,
  fpr_f17,
  fpr_f18,
  fpr_f19,
  fpr_f20,
  fpr_f21,
  fpr_f22,
  fpr_f23,
  fpr_f24,
  fpr_f25,
  fpr_f26,
  fpr_f27,
  fpr_f28,
  fpr_f29,
  fpr_f30,
  fpr_f31,
  fpr_fpscr,
  k_last_fpr = fpr_fpscr,

  // Vector Registers.
  k_first_vmx,
  vmx_v0 = k_first_vmx,
  vmx_v1,
  vmx_v2,
  vmx_v3,
  vmx_v4,
  vmx_v5,
  vmx_v6,
  vmx_v7,
  vmx_v8,
  vmx_v9,
  vmx_v10,
  vmx_v11,
  vmx_v12,
  vmx_v13,
  vmx_v14,
  vmx_v15,
  vmx_v16,
  vmx_v17,
  vmx_v18,
  vmx_v19,
  vmx_v20,
  vmx_v21,
  vmx_v22,
  vmx_v23,
  vmx_v24,
  vmx_v25,
  vmx_v26,
  vmx_v27,
  vmx_v28,
  vmx_v29,
  vmx_v30,
  vmx_v31,
  vmx_vrsave,
  vmx_vscr,
  k_last_vmx = vmx_vscr,

  // Register Counts.
  k_num_registers,
  k_num_gpr_registers = k_last_gpr - k_first_gpr + 1,
  k_num_fpr_registers = k_last_fpr - k_first_fpr + 1,
  k_num_vmx_registers = k_last_vmx - k_first_vmx + 1,
};

} // namespace lldb_private::registers::powerpc::id

#endif // LLDB_REGISTER_ARCH_LLDB_POWERPC_REGISTER_ENUMS_H
