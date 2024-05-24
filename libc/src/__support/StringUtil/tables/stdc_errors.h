//===-- Map of C standard error numbers to strings --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_STRINGUTIL_TABLES_STDC_ERRORS_H
#define LLVM_LIBC_SRC___SUPPORT_STRINGUTIL_TABLES_STDC_ERRORS_H

#include "src/__support/StringUtil/message_mapper.h"

#include <errno.h> // For error macros

namespace LIBC_NAMESPACE {

LIBC_INLINE_VAR constexpr const MsgTable<3> STDC_ERRORS = {
    MsgMapping(0, "Success"),
    MsgMapping(EDOM, "Numerical argument out of domain"),
    MsgMapping(ERANGE, "Numerical result out of range"),
};

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC___SUPPORT_STRINGUTIL_TABLES_STDC_ERRORS_H
