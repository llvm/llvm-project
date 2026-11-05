//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// DWARF register numbers for s390x.
///
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_ARCH_LLDB_S390X_DWARF_REGISTER_ENUMS_H
#define LLDB_REGISTER_ARCH_LLDB_S390X_DWARF_REGISTER_ENUMS_H

#include <cstdint>

namespace lldb_private::registers::s390x::dwarf {

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

  // Floating Point Registers / Vector Registers 0-15.
  f0 = 16,
  f2,
  f4,
  f6,
  f1,
  f3,
  f5,
  f7,
  f8,
  f10,
  f12,
  f14,
  f9,
  f11,
  f13,
  f15,

  // Access Registers.
  acr0 = 48,
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
  pswm = 64,
  pswa,

  // Vector Registers 16-31.
  v16 = 68,
  v18,
  v20,
  v22,
  v17,
  v19,
  v21,
  v23,
  v24,
  v26,
  v28,
  v30,
  v25,
  v27,
  v29,
  v31,
};

} // namespace lldb_private::registers::s390x::dwarf

#endif // LLDB_REGISTER_ARCH_LLDB_S390X_DWARF_REGISTER_ENUMS_H
