//===-- Map from signal numbers to strings in the c std ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_STRING_UTIL_TABLES_STDC_SIGNAL_TABLE_H
#define LLVM_LIBC_SRC_SUPPORT_STRING_UTIL_TABLES_STDC_SIGNAL_TABLE_H

#include <signal.h>

#include "src/__support/StringUtil/message_mapper.h"

namespace __llvm_libc::internal {

inline constexpr const MsgTable<6> STDC_SIGNALS = {
    MsgMapping(SIGINT, "Interrupt"),
    MsgMapping(SIGILL, "Illegal instruction"),
    MsgMapping(SIGABRT, "Aborted"),
    MsgMapping(SIGFPE, "Floating point exception"),
    MsgMapping(SIGSEGV, "Segmentation fault"),
    MsgMapping(SIGTERM, "Terminated"),
};

} // namespace __llvm_libc::internal

#endif // LLVM_LIBC_SRC_SUPPORT_STRING_UTIL_TABLES_LINUX_SIGNAL_TABLE_H
