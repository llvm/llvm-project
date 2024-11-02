//===-- Implementation of sscanf --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/sscanf.h"

#include "src/__support/arg_list.h"
#include "src/stdio/scanf_core/reader.h"
#include "src/stdio/scanf_core/scanf_main.h"
#include "src/stdio/scanf_core/string_reader.h"

#include <stdarg.h>
#include <stdio.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, sscanf,
                   (const char *__restrict buffer,
                    const char *__restrict format, ...)) {
  va_list vlist;
  va_start(vlist, format);
  internal::ArgList args(vlist); // This holder class allows for easier copying
                                 // and pointer semantics, as well as handling
                                 // destruction automatically.
  va_end(vlist);
  scanf_core::StringReader string_reader(buffer);
  scanf_core::Reader reader(&string_reader);
  int ret_val = scanf_core::scanf_main(&reader, format, args);
  // This is done to avoid including stdio.h in the internals. On most systems
  // EOF is -1, so this will be transformed into just "return ret_val".
  return (ret_val == -1) ? EOF : ret_val;
}

} // namespace __llvm_libc
