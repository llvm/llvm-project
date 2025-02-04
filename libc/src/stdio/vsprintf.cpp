//===-- Implementation of vsprintf ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/vsprintf.h"

#include "src/__support/CPP/limits.h"
#include "src/__support/arg_list.h"
#include "src/__support/macros/config.h"
#include "src/stdio/printf_core/printf_main.h"
#include "src/stdio/printf_core/writer.h"

#include <stdarg.h>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, vsprintf,
                   (char *__restrict buffer, const char *__restrict format,
                    va_list vlist)) {
  internal::ArgList args(vlist); // This holder class allows for easier copying
                                 // and pointer semantics, as well as handling
                                 // destruction automatically.

  printf_core::WriteBuffer wb(buffer, cpp::numeric_limits<size_t>::max());
  printf_core::Writer writer(&wb);

  int ret_val = printf_core::printf_main(&writer, format, args);
  wb.buff[wb.buff_cur] = '\0';
  return ret_val;
}

} // namespace LIBC_NAMESPACE_DECL
