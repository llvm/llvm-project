//===-- GPU Implementation of vprintf -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/vprintf.h"

#include "src/__support/CPP/string_view.h"
#include "src/__support/arg_list.h"
#include "src/errno/libc_errno.h"
#include "src/stdio/gpu/vfprintf_utils.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(int, vprintf,
                   (const char *__restrict format, va_list vlist)) {
  cpp::string_view str_view(format);
  int ret_val = vfprintf_internal(stdout, format, str_view.size() + 1, vlist);
  return ret_val;
}

} // namespace LIBC_NAMESPACE
