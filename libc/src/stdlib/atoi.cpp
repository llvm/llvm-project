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

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, atoi, (const char *str)) {
  auto result = internal::strtointeger<int>(str, 10);
  if (result.has_error())
    errno = result.error;

  return result;
}

} // namespace __llvm_libc
