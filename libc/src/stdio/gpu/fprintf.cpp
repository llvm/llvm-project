//===-- GPU Implementation of fprintf -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/fprintf.h"

#include "src/__support/CPP/string_view.h"
#include "src/__support/arg_list.h"
#include "src/errno/libc_errno.h"
#include "src/stdio/gpu/vfprintf_utils.h"

#include <stdio.h>

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(int, fprintf,
                   (::FILE *__restrict stream, const char *__restrict format,
                    ...)) {
  va_list vlist;
  va_start(vlist, format);
  cpp::string_view str_view(format);
  int ret_val = vfprintf_internal(stream, format, str_view.size() + 1, vlist);
  va_end(vlist);
  return ret_val;
}

} // namespace LIBC_NAMESPACE
