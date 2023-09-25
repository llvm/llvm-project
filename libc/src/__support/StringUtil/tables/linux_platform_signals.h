//===-- Map of error numbers to strings for the Linux platform --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_STRING_UTIL_TABLES_LINUX_PLATFORM_SIGNALS_H
#define LLVM_LIBC_SRC___SUPPORT_STRING_UTIL_TABLES_LINUX_PLATFORM_SIGNALS_H

#include "linux_extension_signals.h"
#include "posix_signals.h"
#include "stdc_signals.h"

namespace __llvm_libc {

LIBC_INLINE_VAR constexpr auto PLATFORM_SIGNALS =
    STDC_SIGNALS + POSIX_SIGNALS + LINUX_SIGNALS;

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC___SUPPORT_STRING_UTIL_TABLES_LINUX_PLATFORM_SIGNALS_H
