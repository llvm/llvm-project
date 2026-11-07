//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// EH-frame register numbers for ppc64le.
///
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_ARCH_LLDB_PPC64LE_EHFRAME_REGISTER_ENUMS_H
#define LLDB_REGISTER_ARCH_LLDB_PPC64LE_EHFRAME_REGISTER_ENUMS_H

#include "lldb/Register/Arch/lldb-ppc64le-dwarf-register-enums.h"

namespace lldb_private::registers::ppc64le {

// EH-frame register numbers (eRegisterKindEHFrame)
// ppc64le uses the same DWARF and EH-frame register numbers.
namespace ehframe = dwarf;

} // namespace lldb_private::registers::ppc64le

#endif // LLDB_REGISTER_ARCH_LLDB_PPC64LE_EHFRAME_REGISTER_ENUMS_H
