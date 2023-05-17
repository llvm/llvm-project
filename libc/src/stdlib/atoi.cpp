//===-- Implementation of atoi --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/atoi.h"
#include "src/__support/common.h"
#include "src/__support/str_to_integer.h"
#include "src/errno/libc_errno.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, atoi, (const char *str)) {
  // This is done because the standard specifies that atoi is identical to
  // (int)(strtol).
  auto result = internal::strtointeger<long>(str, 10);
  if (result.has_error())
    libc_errno = result.error;

  return static_cast<int>(result);
}

} // namespace __llvm_libc
