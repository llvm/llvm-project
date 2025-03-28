//===-- Implementation of vscanf --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/vscanf.h"

#include "hdr/stdio_macros.h"
#include "src/__support/OSUtil/io.h"
#include "src/__support/arg_list.h"
#include "src/__support/macros/config.h"
#include "src/stdio/baremetal/scanf_internal.h"
#include "src/stdio/scanf_core/scanf_main.h"

#include <stdarg.h>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, vscanf,
                   (const char *__restrict format, va_list vlist)) {
  internal::ArgList args(vlist); // This holder class allows for easier copying
                                 // and pointer semantics, as well as handling
                                 // destruction automatically.
  va_end(vlist);

  scanf_core::StdinReader reader;
  int retval = scanf_core::scanf_main(&reader, format, args);
  // This is done to avoid including stdio.h in the internals. On most systems
  // EOF is -1, so this will be transformed into just "return retval".
  return (retval == -1) ? EOF : retval;
}

} // namespace LIBC_NAMESPACE_DECL
