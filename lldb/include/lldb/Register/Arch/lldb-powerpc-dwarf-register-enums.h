//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// DWARF register numbers for powerpc.
///
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_ARCH_LLDB_POWERPC_DWARF_REGISTER_ENUMS_H
#define LLDB_REGISTER_ARCH_LLDB_POWERPC_DWARF_REGISTER_ENUMS_H

#include <cstdint>

namespace lldb_private::registers::powerpc::dwarf {

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
  r28,
  r29,
  r30,
  r31,

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

  // Control and Status Registers.
  cr,

  // Floating Point Control and Status Registers.
  fpscr,

  // Control and Status Registers.
  msr,
  vscr,
  xer = 101,

  // General Purpose Registers.
  lr = 108,
  ctr,
  pc,

  // Control and Status Registers.
  vrsave = 356,

  // Vector Registers.
  v0 = 1124,
  v1,
  v2,
  v3,
  v4,
  v5,
  v6,
  v7,
  v8,
  v9,
  v10,
  v11,
  v12,
  v13,
  v14,
  v15,
  v16,
  v17,
  v18,
  v19,
  v20,
  v21,
  v22,
  v23,
  v24,
  v25,
  v26,
  v27,
  v28,
  v29,
  v30,
  v31,
};

} // namespace lldb_private::registers::powerpc::dwarf

#endif // LLDB_REGISTER_ARCH_LLDB_POWERPC_DWARF_REGISTER_ENUMS_H
