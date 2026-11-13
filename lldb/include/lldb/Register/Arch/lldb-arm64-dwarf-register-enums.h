//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// DWARF register numbers for arm64.
///
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_ARCH_LLDB_ARM64_DWARF_REGISTER_ENUMS_H
#define LLDB_REGISTER_ARCH_LLDB_ARM64_DWARF_REGISTER_ENUMS_H

#include <cstdint>

namespace lldb_private::registers::arm64::dwarf {

// DWARF register numbers (eRegisterKindDWARF)
enum : uint32_t {
  // General Purpose Registers.
  x0 = 0,
  x1,
  x2,
  x3,
  x4,
  x5,
  x6,
  x7,
  x8,
  x9,
  x10,
  x11,
  x12,
  x13,
  x14,
  x15,
  x16,
  x17,
  x18,
  x19,
  x20,
  x21,
  x22,
  x23,
  x24,
  x25,
  x26,
  x27,
  x28,
  x29 = 29,
  fp = x29,
  x30 = 30,
  lr = x30,
  x31 = 31,
  sp = x31,
  pc = 32,

  // Status Registers.
  elr_mode = 33,
  ra_sign_state = 34,

  // Thread-Local Storage Registers.
  tpidrr0_el0 = 35,
  tpidr_el0 = 36,
  tpidr_el1 = 37,
  tpidr_el2 = 38,
  tpidr_el3 = 39,

  // Register numbers 40-45 are reserved.

  // SVE Vector Granule Count.
  vg = 46,

  // SVE First-Fault Register (VG * 8 bits).
  ffr = 47,

  // SVE Predicate Registers (VG * 8 bits).
  p0 = 48,
  p1,
  p2,
  p3,
  p4,
  p5,
  p6,
  p7,
  p8,
  p9,
  p10,
  p11,
  p12,
  p13,
  p14,
  p15,

  // 128-bit Vector Registers.
  v0 = 64,
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

  // SVE Vector Registers (VG * 64 bits).
  z0 = 96,
  z1,
  z2,
  z3,
  z4,
  z5,
  z6,
  z7,
  z8,
  z9,
  z10,
  z11,
  z12,
  z13,
  z14,
  z15,
  z16,
  z17,
  z18,
  z19,
  z20,
  z21,
  z22,
  z23,
  z24,
  z25,
  z26,
  z27,
  z28,
  z29,
  z30,
  z31
};

} // namespace lldb_private::registers::arm64::dwarf

#endif // LLDB_REGISTER_ARCH_LLDB_ARM64_DWARF_REGISTER_ENUMS_H
