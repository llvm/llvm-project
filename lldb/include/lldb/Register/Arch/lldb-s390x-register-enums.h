//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// LLDB register IDs for s390x.
///
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_ARCH_LLDB_S390X_REGISTER_ENUMS_H
#define LLDB_REGISTER_ARCH_LLDB_S390X_REGISTER_ENUMS_H

#include <cstdint>

namespace lldb_private::registers::s390x::id {

// LLDB register codes (e.g. RegisterKind == eRegisterKindLLDB)
enum : uint32_t {
  // General Purpose Registers.
  k_first_gpr,
  r0 = k_first_gpr,
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

  // Access Registers.
  acr0,
  acr1,
  acr2,
  acr3,
  acr4,
  acr5,
  acr6,
  acr7,
  acr8,
  acr9,
  acr10,
  acr11,
  acr12,
  acr13,
  acr14,
  acr15,

  // Program Status Word Registers.
  pswm,
  pswa,
  k_last_gpr = pswa,

  // Floating Point Registers.
  k_first_fpr,
  f0 = k_first_fpr,
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

  // Floating Point Control and Status Registers.
  fpc,
  k_last_fpr = fpc,

  // Linux-Specific Registers.
  // These are only available on Linux.
  k_first_linux,
  orig_r2 = k_first_linux,
  last_break,
  system_call,
  k_last_linux = system_call,

  // Register Counts.
  k_num_registers,
  k_num_gpr_registers = k_last_gpr - k_first_gpr + 1,
  k_num_fpr_registers = k_last_fpr - k_first_fpr + 1,
  k_num_linux_registers = k_last_linux - k_first_linux + 1,
  k_num_user_registers = k_num_gpr_registers + k_num_fpr_registers,
};

} // namespace lldb_private::registers::s390x::id

#endif // LLDB_REGISTER_ARCH_LLDB_S390X_REGISTER_ENUMS_H
