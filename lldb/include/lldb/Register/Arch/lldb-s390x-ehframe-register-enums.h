//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// EH-frame register numbers for s390x.
///
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_ARCH_LLDB_S390X_EHFRAME_REGISTER_ENUMS_H
#define LLDB_REGISTER_ARCH_LLDB_S390X_EHFRAME_REGISTER_ENUMS_H

#include "lldb/Register/Arch/lldb-s390x-dwarf-register-enums.h"

namespace lldb_private::registers::s390x {

// EH-frame register numbers (eRegisterKindEHFrame)
// s390x uses the same DWARF and EH-frame register numbers.
namespace ehframe = dwarf;

} // namespace lldb_private::registers::s390x

#endif // LLDB_REGISTER_ARCH_LLDB_S390X_EHFRAME_REGISTER_ENUMS_H
