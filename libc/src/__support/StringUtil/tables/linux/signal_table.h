//===-- Map from signal numbers to strings on linux -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_STRING_UTIL_TABLES_LINUX_SIGNAL_TABLE_H
#define LLVM_LIBC_SRC_SUPPORT_STRING_UTIL_TABLES_LINUX_SIGNAL_TABLE_H

#include <signal.h>

#include "src/__support/StringUtil/message_mapper.h"

namespace __llvm_libc::internal {

// The array being larger than necessary isn't a problem. The MsgMappings will
// be set to their default state which maps 0 to an empty string. This will get
// filtered out in the MessageMapper building stage.
inline constexpr const MsgTable<3> LINUX_SIGNALS = {
#ifdef SIGSTKFLT
    MsgMapping(SIGSTKFLT, "Stack fault"), // unused
#endif
    MsgMapping(SIGWINCH, "Window changed"),
#ifdef SIGPWR
    MsgMapping(SIGPWR, "Power failure"), // ignored
#endif
};

} // namespace __llvm_libc::internal

#endif // LLVM_LIBC_SRC_SUPPORT_STRING_UTIL_TABLES_LINUX_SIGNAL_TABLE_H
