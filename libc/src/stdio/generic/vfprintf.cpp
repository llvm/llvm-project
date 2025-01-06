//===-- Implementation of vfprintf ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/vfprintf.h"

#include "src/__support/File/file.h"
#include "src/__support/arg_list.h"
#include "src/__support/macros/config.h"
#include "src/stdio/printf_core/vfprintf_internal.h"

#include "hdr/types/FILE.h"
#include <stdarg.h>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, vfprintf,
                   (::FILE *__restrict stream, const char *__restrict format,
                    va_list vlist)) {
  internal::ArgList args(vlist); // This holder class allows for easier copying
                                 // and pointer semantics, as well as handling
                                 // destruction automatically.
  int ret_val = printf_core::vfprintf_internal(stream, format, args);
  return ret_val;
}

} // namespace LIBC_NAMESPACE_DECL
