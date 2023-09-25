//===-- Map of C standard error numbers to strings --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_STRING_UTIL_TABLES_STDC_ERRORS_H
#define LLVM_LIBC_SRC___SUPPORT_STRING_UTIL_TABLES_STDC_ERRORS_H

#include "src/__support/StringUtil/message_mapper.h"

#include <errno.h> // For error macros

namespace __llvm_libc {

LIBC_INLINE_VAR constexpr const MsgTable<4> STDC_ERRORS = {
    MsgMapping(0, "Success"),
    MsgMapping(EDOM, "Numerical argument out of domain"),
    MsgMapping(ERANGE, "Numerical result out of range"),
    MsgMapping(EILSEQ, "Invalid or incomplete multibyte or wide character"),
};

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC___SUPPORT_STRING_UTIL_TABLES_LINUX_ERRORS_H
