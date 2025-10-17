//===-- Implementation of asprintf -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/asprintf.h"
#include "hdr/errno_macros.h"
#include "src/__support/arg_list.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/stdio/printf_core/vasprintf_internal.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, asprintf,
                   (char **__restrict buffer, const char *__restrict format,
                    ...)) {
  va_list vlist;
  va_start(vlist, format);
  internal::ArgList args(vlist); // This holder class allows for easier copying
                                 // and pointer semantics, as well as handling
                                 // destruction automatically.
  va_end(vlist);
  auto ret_val = printf_core::vasprintf_internal(buffer, format, args);
  if (!ret_val.has_value()) {
    libc_errno = printf_core::internal_error_to_errno(ret_val.error());
    return -1;
  }
  if (ret_val.value() > cpp::numeric_limits<int>::max()) {
    libc_errno = EOVERFLOW;
    return -1;
  }

  return static_cast<int>(ret_val.value());
}

} // namespace LIBC_NAMESPACE_DECL
