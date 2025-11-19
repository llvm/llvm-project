//===-- Implementation of sprintf -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/sprintf.h"

#include "src/__support/CPP/limits.h"
#include "src/__support/arg_list.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/stdio/printf_core/core_structs.h"
#include "src/stdio/printf_core/error_mapper.h"
#include "src/stdio/printf_core/printf_main.h"
#include "src/stdio/printf_core/writer.h"

#include <stdarg.h>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, sprintf,
                   (char *__restrict buffer, const char *__restrict format,
                    ...)) {
  va_list vlist;
  va_start(vlist, format);
  internal::ArgList args(vlist); // This holder class allows for easier copying
                                 // and pointer semantics, as well as handling
                                 // destruction automatically.
  va_end(vlist);

  printf_core::WriteBuffer<
      printf_core::Mode<printf_core::WriteMode::RESIZE_AND_FILL_BUFF>::value>
      wb(buffer, cpp::numeric_limits<size_t>::max());
  printf_core::Writer writer(wb);

  auto ret_val = printf_core::printf_main(&writer, format, args);
  if (!ret_val.has_value()) {
    libc_errno = printf_core::internal_error_to_errno(ret_val.error());
    return -1;
  }
  wb.buff[wb.buff_cur] = '\0';

  if (ret_val.value() > static_cast<size_t>(cpp::numeric_limits<int>::max())) {
    libc_errno =
        printf_core::internal_error_to_errno(-printf_core::OVERFLOW_ERROR);
    return -1;
  }

  return static_cast<int>(ret_val.value());
}

} // namespace LIBC_NAMESPACE_DECL
