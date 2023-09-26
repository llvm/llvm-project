//===-- Implementation of vprintf -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/vprintf.h"

#include "src/__support/File/file.h"
#include "src/__support/arg_list.h"
#include "src/stdio/printf_core/vfprintf_internal.h"

#include <stdarg.h>
#include <stdio.h>

#ifndef LIBC_COPT_STDIO_USE_SYSTEM_FILE
#define PRINTF_STDOUT LIBC_NAMESPACE::stdout
#else // LIBC_COPT_STDIO_USE_SYSTEM_FILE
#define PRINTF_STDOUT ::stdout
#endif // LIBC_COPT_STDIO_USE_SYSTEM_FILE

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(int, vprintf,
                   (const char *__restrict format, va_list vlist)) {
  internal::ArgList args(vlist); // This holder class allows for easier copying
                                 // and pointer semantics, as well as handling
                                 // destruction automatically.
  int ret_val = printf_core::vfprintf_internal(
      reinterpret_cast<::FILE *>(PRINTF_STDOUT), format, args);
  return ret_val;
}

} // namespace LIBC_NAMESPACE
