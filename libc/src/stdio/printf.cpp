//===-- Implementation of printf --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/printf.h"

#include "src/__support/File/file.h"
#include "src/__support/arg_list.h"
#include "src/stdio/printf_core/vfprintf_internal.h"

#include <stdarg.h>
#include <stdio.h>

#ifndef LIBC_COPT_PRINTF_USE_SYSTEM_FILE
#define PRINTF_STDOUT __llvm_libc::stdout
#else // LIBC_COPT_PRINTF_USE_SYSTEM_FILE
#define PRINTF_STDOUT ::stdout
#endif // LIBC_COPT_PRINTF_USE_SYSTEM_FILE

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, printf, (const char *__restrict format, ...)) {
  va_list vlist;
  va_start(vlist, format);
  internal::ArgList args(vlist); // This holder class allows for easier copying
                                 // and pointer semantics, as well as handling
                                 // destruction automatically.
  va_end(vlist);
  int ret_val = printf_core::vfprintf_internal(
      reinterpret_cast<::FILE *>(PRINTF_STDOUT), format, args);
  return ret_val;
}

} // namespace __llvm_libc
