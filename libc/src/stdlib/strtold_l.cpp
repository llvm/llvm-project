//===-- Implementation of strtold_l ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/strtold_l.h"
#include "src/stdlib/strtold.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(long double, strtold_l,
                   (const char *__restrict str, char **__restrict str_end, locale_t)) {

  return strtold(str, str_end);
}

} // namespace LIBC_NAMESPACE
