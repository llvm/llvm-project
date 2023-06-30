//===-- Map of error numbers to strings for the Linux platform --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_STRING_UTIL_TABLES_LINUX_PLATFORM_ERRORS_H
#define LLVM_LIBC_SRC_SUPPORT_STRING_UTIL_TABLES_LINUX_PLATFORM_ERRORS_H

#include "linux_extension_errors.h"
#include "posix_errors.h"
#include "stdc_errors.h"

namespace __llvm_libc {

LIBC_INLINE_VAR constexpr auto PLATFORM_ERRORS =
    STDC_ERRORS + POSIX_ERRORS + LINUX_ERRORS;

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_STRING_UTIL_TABLES_LINUX_PLATFORM_ERRORS_H
