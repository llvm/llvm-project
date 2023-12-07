//===-- Implementation of atof --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/atof.h"
#include "src/__support/common.h"
#include "src/__support/str_to_float.h"
#include "src/errno/libc_errno.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(double, atof, (const char *str)) {
  auto result = internal::strtofloatingpoint<double>(str);
  if (result.has_error())
    libc_errno = result.error;

  return result.value;
}

} // namespace __llvm_libc
