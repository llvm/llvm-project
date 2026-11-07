//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// EH-frame register numbers for riscv.
///
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_ARCH_LLDB_RISCV_EHFRAME_REGISTER_ENUMS_H
#define LLDB_REGISTER_ARCH_LLDB_RISCV_EHFRAME_REGISTER_ENUMS_H

#include "lldb/Register/Arch/lldb-riscv-dwarf-register-enums.h"

namespace lldb_private::registers::riscv {

// EH-frame register numbers (eRegisterKindEHFrame)
// riscv uses the same DWARF and EH-frame register numbers.
namespace ehframe = dwarf;

} // namespace lldb_private::registers::riscv

#endif // LLDB_REGISTER_ARCH_LLDB_RISCV_EHFRAME_REGISTER_ENUMS_H
