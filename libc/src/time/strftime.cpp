//===-- Implementation of strftime function -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/strftime.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/errno/libc_errno.h"
#include "src/time/time_utils.h"

#include "src/stdio/printf_core/writer.h"
#include "src/time/strftime_core/strftime_main.h"
namespace LIBC_NAMESPACE_DECL {

size_t strftime(char *__restrict buffer, size_t buffsz,
                const char *__restrict format, const struct tm *timeptr) {

  printf_core::WriteBuffer wb(buffer, (buffsz > 0 ? buffsz - 1 : 0),
                              strftime_core::overflow_write_mock, nullptr);
  printf_core::Writer writer(&wb);
  int ret = strftime_core::strftime_main(&writer, format, timeptr);
  if (buffsz > 0) // if the buffsz is 0 the buffer may be a null pointer.
    wb.buff[wb.buff_cur] = '\0';
  return ret > 0 ? ret : 0;
}

} // namespace LIBC_NAMESPACE_DECL
