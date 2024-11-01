//===-- Implementation of asprintf -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/asprintf.h"
#include "src/__support/arg_list.h"
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
  int ret = printf_core::vasprintf_internal(buffer, format, args);
  return ret;
}

} // namespace LIBC_NAMESPACE_DECL
