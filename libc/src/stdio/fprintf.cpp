//===-- Implementation of fprintf -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/fprintf.h"

#include "src/__support/arg_list.h"
#include "src/stdio/ferror.h"
#include "src/stdio/printf_core/file_writer.h"
#include "src/stdio/printf_core/printf_main.h"
#include "src/stdio/printf_core/writer.h"

#include <stdarg.h>
#include <stdio.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, fprintf,
                   (::FILE *__restrict stream, const char *__restrict format,
                    ...)) {
  va_list vlist;
  va_start(vlist, format);
  internal::ArgList args(vlist); // This holder class allows for easier copying
                                 // and pointer semantics, as well as handling
                                 // destruction automatically.
  va_end(vlist);
  printf_core::Writer writer(reinterpret_cast<void *>(stream),
                             printf_core::write_to_file);

  int ret_val = printf_core::printf_main(&writer, format, args);
  if (__llvm_libc::ferror(stream))
    return -1;
  return ret_val;
}

} // namespace __llvm_libc
