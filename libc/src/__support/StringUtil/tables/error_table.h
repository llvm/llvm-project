//===-- Map from error numbers to strings -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_STRING_UTIL_TABLES_ERROR_TABLE_H
#define LLVM_LIBC_SRC_SUPPORT_STRING_UTIL_TABLES_ERROR_TABLE_H

#include "src/__support/StringUtil/message_mapper.h"

#include "posix_error_table.h"
#include "stdc_error_table.h"

#ifdef __linux__
#include "linux/error_table.h"
#endif

namespace __llvm_libc::internal {

#ifdef __linux__
inline constexpr auto PLATFORM_ERRORS =
    STDC_ERRORS + POSIX_ERRORS + LINUX_ERRORS;
#else
inline constexpr auto PLATFORM_ERRORS = STDC_ERRORS;
#endif

} // namespace __llvm_libc::internal

#endif // LLVM_LIBC_SRC_SUPPORT_STRING_UTIL_TABLES_ERROR_TABLE_H
