//===-- Linux implementation of error converter -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_PRINTF_CORE_LINUX_ERROR_CONVERTER_H
#define LLVM_LIBC_SRC_STDIO_PRINTF_CORE_LINUX_ERROR_CONVERTER_H

#include "hdr/errno_macros.h"
#include "src/stdio/printf_core/core_structs.h"
#include "src/stdio/printf_core/error_converter.h"

namespace LIBC_NAMESPACE_DECL {
namespace printf_core {

LIBC_INLINE static int internal_error_to_errno(int internal_error) {
  // System error occured, return error as is.
  if (internal_error < 1001 && internal_error > 0) {
    return internal_error;
  }

  // Map internal error to POSIX errnos.
  switch (-internal_error) {
  case WRITE_OK:
    return 0;
  case FILE_WRITE_ERROR:
    return EIO;
  case FILE_STATUS_ERROR:
    return EIO;
  case NULLPTR_WRITE_ERROR:
    return EINVAL;
  case INT_CONVERSION_ERROR:
    return ERANGE;
  case FIXED_POINT_CONVERSION_ERROR:
    return EINVAL;
  case ALLOCATION_ERROR:
    return ENOMEM;
  case OVERFLOW_ERROR:
    return EOVERFLOW;
  default:
    LIBC_ASSERT(
        false &&
        "Invalid internal printf error code passed to internal_error_to_errno");
    return EINVAL;
  }
}

} // namespace printf_core
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDIO_PRINTF_CORE_LINUX_ERROR_CONVERTER_H
