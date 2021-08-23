//===-- Implementation of atoi --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/atoi.h"
#include "src/__support/common.h"
#include "src/__support/str_conv_utils.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, atoi, (const char *str)) {
  return internal::strtointeger<int>(str, nullptr, 10);
}

} // namespace __llvm_libc
