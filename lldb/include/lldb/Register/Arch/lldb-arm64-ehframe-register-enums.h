//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// EH-frame register numbers for arm64.
///
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_ARCH_LLDB_ARM64_EHFRAME_REGISTER_ENUMS_H
#define LLDB_REGISTER_ARCH_LLDB_ARM64_EHFRAME_REGISTER_ENUMS_H

#include "lldb/Register/Arch/lldb-arm64-dwarf-register-enums.h"

namespace lldb_private::registers::arm64 {

// EH-frame register numbers (eRegisterKindEHFrame)
// arm64 uses the same DWARF and EH-frame register numbers.
namespace ehframe = dwarf;

} // namespace lldb_private::registers::arm64

#endif // LLDB_REGISTER_ARCH_LLDB_ARM64_EHFRAME_REGISTER_ENUMS_H
