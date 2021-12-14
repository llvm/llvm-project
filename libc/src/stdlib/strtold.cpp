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

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(long double, strtold,
                   (const char *__restrict str, char **__restrict str_end)) {
  return internal::strtofloatingpoint<long double>(str, str_end);
}

} // namespace __llvm_libc
