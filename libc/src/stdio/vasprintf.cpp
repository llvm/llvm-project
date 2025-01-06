//===-- Implementation of vasprintf -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/vasprintf.h"
#include "src/__support/arg_list.h"
#include "src/stdio/printf_core/vasprintf_internal.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, vasprintf,
                   (char **__restrict ret, const char *__restrict format,
                    va_list vlist)) {
  internal::ArgList args(vlist); // This holder class allows for easier copying
                                 // and pointer semantics, as well as handling
                                 // destruction automatically.
  return printf_core::vasprintf_internal(ret, format, args);
}

} // namespace LIBC_NAMESPACE_DECL
