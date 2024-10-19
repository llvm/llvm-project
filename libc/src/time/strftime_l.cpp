//===-- Implementation of strftime_l function -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/strftime_l.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/errno/libc_errno.h"
#include "src/time/time_utils.h"

#include "src/stdio/printf_core/writer.h"
#include "src/time/strftime_core/strftime_main.h"
namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(size_t, strftime_l,
                   (char *__restrict buffer, size_t buffsz,
                    const char *__restrict format, const struct tm *timeptr,
                    locale_t)) {
  printf_core::WriteBuffer wb(buffer, (buffsz > 0 ? buffsz - 1 : 0));
  printf_core::Writer writer(&wb);
  strftime_core::strftime_main(&writer, format, timeptr);
  return writer.get_chars_written();
}

} // namespace LIBC_NAMESPACE_DECL
