//===-- Implementation of strtoull ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/strtoull.h"
#include "src/__support/common.h"
#include "src/__support/str_to_integer.h"
#include "src/errno/libc_errno.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(unsigned long long, strtoull,
                   (const char *__restrict str, char **__restrict str_end,
                    int base)) {
  auto result = internal::strtointeger<unsigned long long>(str, base);
  if (result.has_error())
    libc_errno = result.error;

  if (str_end != nullptr)
    *str_end = const_cast<char *>(str + result.parsed_len);

  return result;
}

} // namespace __llvm_libc
