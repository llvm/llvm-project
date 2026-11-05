//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// EH-frame register numbers for ppc64be.
///
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_ARCH_LLDB_PPC64BE_EHFRAME_REGISTER_ENUMS_H
#define LLDB_REGISTER_ARCH_LLDB_PPC64BE_EHFRAME_REGISTER_ENUMS_H

#include "lldb/Register/Arch/lldb-ppc64-ehframe-register-enums.h"

namespace lldb_private::registers::ppc64be {

namespace ehframe {
namespace aix = lldb_private::registers::ppc64::ehframe::aix;
namespace elf = lldb_private::registers::ppc64::ehframe::elf;
} // namespace ehframe

} // namespace lldb_private::registers::ppc64be

#endif
