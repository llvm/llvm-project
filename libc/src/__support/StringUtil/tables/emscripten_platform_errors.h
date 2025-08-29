//===-- Map of error numbers to strings for the Emscripten platform --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_STRINGUTIL_TABLES_EMSCRPITEN_PLATFORM_ERRORS_H
#define LLVM_LIBC_SRC___SUPPORT_STRINGUTIL_TABLES_EMSCRPITEN_PLATFORM_ERRORS_H

#include "posix_errors.h"
#include "src/__support/macros/config.h"
#include "stdc_errors.h"

namespace LIBC_NAMESPACE_DECL {

LIBC_INLINE_VAR constexpr auto PLATFORM_ERRORS =
    STDC_ERRORS + POSIX_ERRORS;

LIBC_INLINE_VAR constexpr auto PLATFORM_ERRNO_NAMES =
    STDC_ERRNO_NAMES + POSIX_ERRNO_NAMES;

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_STRINGUTIL_TABLES_EMSCRPITEN_PLATFORM_ERRORS_H
