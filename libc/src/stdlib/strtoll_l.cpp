//===-- Implementation of strtoll_l ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/strtoll_l.h"
#include "src/stdlib/strtoll.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(long long, strtoll_l,
                   (const char *__restrict str, char **__restrict str_end,
                    int base, locale_t)) {
 
  return strtoll(str, str_end, base);
}

} // namespace LIBC_NAMESPACE
