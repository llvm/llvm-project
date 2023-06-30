//===-- Map of C standard signal numbers to strings -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_STRING_UTIL_TABLES_STDC_SIGNALS_H
#define LLVM_LIBC_SRC_SUPPORT_STRING_UTIL_TABLES_STDC_SIGNALS_H

#include <signal.h> // For signal numbers

#include "src/__support/StringUtil/message_mapper.h"

namespace __llvm_libc {

LIBC_INLINE_VAR constexpr const MsgTable<6> STDC_SIGNALS = {
    MsgMapping(SIGINT, "Interrupt"),
    MsgMapping(SIGILL, "Illegal instruction"),
    MsgMapping(SIGABRT, "Aborted"),
    MsgMapping(SIGFPE, "Floating point exception"),
    MsgMapping(SIGSEGV, "Segmentation fault"),
    MsgMapping(SIGTERM, "Terminated"),
};

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_STRING_UTIL_TABLES_LINUX_SIGNALS_H
