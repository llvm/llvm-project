//===-- Implementation of sscanf --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/sscanf.h"

#include "src/__support/CPP/limits.h"
#include "src/__support/arg_list.h"
#include "src/stdio/scanf_core/reader.h"
#include "src/stdio/scanf_core/scanf_main.h"

#include <stdarg.h>
#include <stdio.h>

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(int, sscanf,
                   (const char *__restrict buffer,
                    const char *__restrict format, ...)) {
  va_list vlist;
  va_start(vlist, format);
  internal::ArgList args(vlist); // This holder class allows for easier copying
                                 // and pointer semantics, as well as handling
                                 // destruction automatically.
  va_end(vlist);
  scanf_core::ReadBuffer rb{const_cast<char *>(buffer),
                            cpp::numeric_limits<size_t>::max()};
  scanf_core::Reader reader(&rb);
  int ret_val = scanf_core::scanf_main(&reader, format, args);
  // This is done to avoid including stdio.h in the internals. On most systems
  // EOF is -1, so this will be transformed into just "return ret_val".
  return (ret_val == -1) ? EOF : ret_val;
}

} // namespace LIBC_NAMESPACE
