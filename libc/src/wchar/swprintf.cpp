//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the implementation of the swprintf function.
///
//===----------------------------------------------------------------------===//

#include "src/wchar/swprintf.h"

#include "hdr/errno_macros.h"
#include "hdr/types/size_t.h"
#include "hdr/types/wchar_t.h"
#include "src/__support/CPP/limits.h"
#include "src/__support/arg_list.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/printf_core/core_structs.h"
#include "src/__support/printf_core/error_mapper.h"
#include "src/__support/printf_core/printf_main.h"
#include "src/__support/printf_core/writer.h"

#include <stdarg.h>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, swprintf,
                   (wchar_t *__restrict buffer, size_t bufsz,
                    const wchar_t *__restrict format, ...)) {
  va_list vlist;
  va_start(vlist, format);
  internal::ArgList args(vlist); // This holder class allows for easier copying
                                 // and pointer semantics, as well as handling
                                 // destruction automatically.
  va_end(vlist);

  printf_core::Writer writer = printf_core::make_drop_overflow_writer(
      buffer, (bufsz > 0 ? bufsz - 1 : 0));

#ifdef LIBC_COPT_PRINTF_MODULAR
  LIBC_INLINE_ASM(".reloc ., BFD_RELOC_NONE, __printf_float");
  auto ret_val = printf_core::printf_main_modular(&writer, format, args);
#else
  auto ret_val = printf_core::printf_main(&writer, format, args);
#endif
  if (!ret_val.has_value()) {
    libc_errno = printf_core::internal_error_to_errno(ret_val.error());
    return -1;
  }
  if (bufsz > 0) { // if the bufsz is 0 the buffer may be a null pointer.
    printf_core::WriteBuffer<wchar_t> &wb = writer.get_write_buffer();
    wb.buff[wb.buff_cur] = L'\0';
  }

  if (ret_val.value() > static_cast<size_t>(cpp::numeric_limits<int>::max())) {
    libc_errno =
        printf_core::internal_error_to_errno(-printf_core::OVERFLOW_ERROR);
    return -1;
  }

  return static_cast<int>(ret_val.value());
}

} // namespace LIBC_NAMESPACE_DECL
