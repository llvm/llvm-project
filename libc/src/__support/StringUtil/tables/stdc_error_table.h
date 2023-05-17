//===-- Map from error numbers to strings in the c std ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_STRING_UTIL_TABLES_STDC_ERROR_TABLE_H
#define LLVM_LIBC_SRC_SUPPORT_STRING_UTIL_TABLES_STDC_ERROR_TABLE_H

#include "src/errno/libc_errno.h" // For error macros

#include "src/__support/StringUtil/message_mapper.h"

namespace __llvm_libc::internal {

inline constexpr const MsgTable<4> STDC_ERRORS = {
    MsgMapping(0, "Success"),
    MsgMapping(EDOM, "Numerical argument out of domain"),
    MsgMapping(ERANGE, "Numerical result out of range"),
    MsgMapping(EILSEQ, "Invalid or incomplete multibyte or wide character"),
};

} // namespace __llvm_libc::internal

#endif // LLVM_LIBC_SRC_SUPPORT_STRING_UTIL_TABLES_LINUX_ERROR_TABLE_H
