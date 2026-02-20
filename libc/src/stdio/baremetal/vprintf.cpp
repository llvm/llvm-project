//===-- Implementation of vprintf -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/vprintf.h"

#include "hdr/stdio_macros.h"
#include "src/__support/arg_list.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/stdio/baremetal/vfprintf_internal.h"

#include <stdarg.h>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, vprintf,
                   (const char *__restrict format, va_list vlist)) {
  internal::ArgList args(vlist); // This holder class allows for easier copying
                                 // and pointer semantics, as well as handling
                                 // destruction automatically.

#ifdef LIBC_COPT_PRINTF_MODULAR
  LIBC_INLINE_ASM(".reloc ., BFD_RELOC_NONE, __printf_float");
  return vfprintf_internal<true>(stdout, format, args);
#else
  return vfprintf_internal(stdout, format, args);
#endif
}

} // namespace LIBC_NAMESPACE_DECL
