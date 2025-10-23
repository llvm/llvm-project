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
#include "src/__support/macros/config.h"
#include "src/stdio/printf_core/core_structs.h"
#include "src/stdio/printf_core/error_converter.h"
#include "src/stdio/printf_core/vfprintf_internal.h"

#include "hdr/types/FILE.h"
#include <stdarg.h>

#ifndef LIBC_COPT_STDIO_USE_SYSTEM_FILE
#define PRINTF_STDOUT LIBC_NAMESPACE::stdout
#else // LIBC_COPT_STDIO_USE_SYSTEM_FILE
#define PRINTF_STDOUT ::stdout
#endif // LIBC_COPT_STDIO_USE_SYSTEM_FILE

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, printf, (const char *__restrict format, ...)) {
  va_list vlist;
  va_start(vlist, format);
  internal::ArgList args(vlist); // This holder class allows for easier copying
                                 // and pointer semantics, as well as handling
                                 // destruction automatically.
  va_end(vlist);
  auto ret_val = printf_core::vfprintf_internal(
      reinterpret_cast<::FILE *>(PRINTF_STDOUT), format, args);
  if (!ret_val.has_value()) {
    libc_errno = printf_core::internal_error_to_errno(ret_val.error());
    return -1;
  }
  if (ret_val.value() > cpp::numeric_limits<int>::max()) {
    libc_errno =
        printf_core::internal_error_to_errno(-printf_core::OVERFLOW_ERROR);
    return -1;
  }

  return static_cast<int>(ret_val.value());
}

} // namespace LIBC_NAMESPACE_DECL
