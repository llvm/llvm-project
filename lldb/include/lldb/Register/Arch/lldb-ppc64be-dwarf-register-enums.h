//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// DWARF register numbers for ppc64be, selected by convention.
///
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_ARCH_LLDB_PPC64BE_DWARF_REGISTER_ENUMS_H
#define LLDB_REGISTER_ARCH_LLDB_PPC64BE_DWARF_REGISTER_ENUMS_H

#include "lldb/Register/Arch/lldb-ppc64-dwarf-register-enums.h"

namespace lldb_private::registers::ppc64be {

namespace dwarf {
namespace aix = lldb_private::registers::ppc64::dwarf::aix;
namespace elf = lldb_private::registers::ppc64::dwarf::elf;
} // namespace dwarf

} // namespace lldb_private::registers::ppc64be

#endif
