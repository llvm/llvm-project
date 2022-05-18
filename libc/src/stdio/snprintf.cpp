//===-- Implementation of snprintf ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/snprintf.h"

#include "src/__support/arg_list.h"
#include "src/stdio/printf_core/printf_main.h"
#include "src/stdio/printf_core/string_writer.h"
#include "src/stdio/printf_core/writer.h"

#include <stdarg.h>
#include <stddef.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, snprintf,
                   (char *__restrict buffer, size_t buffsz,
                    const char *__restrict format, ...)) {
  va_list vlist;
  va_start(vlist, format);
  internal::ArgList args(vlist); // This holder class allows for easier copying
                                 // and pointer semantics, as well as handling
                                 // destruction automatically.
  va_end(vlist);
  printf_core::StringWriter str_writer(buffer, (buffsz > 0 ? buffsz - 1 : 0));
  printf_core::Writer writer(reinterpret_cast<void *>(&str_writer),
                             printf_core::write_to_string);

  int ret_val = printf_core::printf_main(&writer, format, args);
  if (buffsz > 0) // if the buffsz is 0 the buffer may be a null pointer.
    str_writer.terminate();
  return ret_val;
}

} // namespace __llvm_libc
