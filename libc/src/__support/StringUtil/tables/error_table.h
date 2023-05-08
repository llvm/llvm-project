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

#if defined(__linux__) || defined(__Fuchsia__)
#define USE_LINUX_PLATFORM_ERRORS 1
#else
#define USE_LINUX_PLATFORM_ERRORS 0
#endif

#if USE_LINUX_PLATFORM_ERRORS
#include "linux/error_table.h"
#endif

namespace __llvm_libc::internal {

inline constexpr auto PLATFORM_ERRORS = []() {
  if constexpr (USE_LINUX_PLATFORM_ERRORS) {
    return STDC_ERRORS + POSIX_ERRORS + LINUX_ERRORS;
  } else {
    return STDC_ERRORS;
  }
}();

} // namespace __llvm_libc::internal

#endif // LLVM_LIBC_SRC_SUPPORT_STRING_UTIL_TABLES_ERROR_TABLE_H
