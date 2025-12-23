//===-- Implementation of strftime function -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/strftime.h"
#include "hdr/types/size_t.h"
#include "hdr/types/struct_tm.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/stdio/printf_core/writer.h"
#include "src/time/strftime_core/strftime_main.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(size_t, strftime,
                   (char *__restrict buffer, size_t buffsz,
                    const char *__restrict format, const tm *timeptr)) {
  printf_core::WriteBuffer<printf_core::Mode<
      printf_core::WriteMode::FILL_BUFF_AND_DROP_OVERFLOW>::value>
      wb(buffer, (buffsz > 0 ? buffsz - 1 : 0));
  printf_core::Writer writer(wb);
  auto ret = strftime_core::strftime_main(&writer, format, timeptr);
  if (buffsz > 0) // if the buffsz is 0 the buffer may be a null pointer.
    wb.buff[wb.buff_cur] = '\0';
  return (!ret.has_value() || ret.value() >= buffsz) ? 0 : ret.value();
}

} // namespace LIBC_NAMESPACE_DECL
