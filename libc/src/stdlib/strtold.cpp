//===-- Implementation of strtold -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/strtold.h"
#include "src/__support/common.h"
#include "src/__support/str_to_float.h"
#include "src/errno/libc_errno.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(long double, strtold,
                   (const char *__restrict str, char **__restrict str_end)) {
  auto result = internal::strtofloatingpoint<long double>(str);
  if (result.has_error())
    libc_errno = result.error;

  if (str_end != NULL)
    *str_end = const_cast<char *>(str + result.parsed_len);

  return result.value;
}

} // namespace __llvm_libc
