//===-- GPU Implementation of vfprintf ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/vfprintf.h"

#include "hdr/types/FILE.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/arg_list.h"
#include "src/errno/libc_errno.h"
#include "src/stdio/gpu/vfprintf_utils.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, vfprintf,
                   (::FILE *__restrict stream, const char *__restrict format,
                    va_list vlist)) {
  cpp::string_view str_view(format);
  int ret_val = vfprintf_internal(stream, format, str_view.size() + 1, vlist);
  return ret_val;
}

} // namespace LIBC_NAMESPACE_DECL
