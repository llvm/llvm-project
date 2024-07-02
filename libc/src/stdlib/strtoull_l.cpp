//===-- Implementation of strtoull_l --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/strtoull_l.h"
#include "src/__support/common.h"
#include "src/stdlib/strtoull.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(unsigned long long, strtoull_l,
                   (const char *__restrict str, char **__restrict str_end,
                    int base, locale_t)) {

  return strtoull(str, str_end, base);
}

} // namespace LIBC_NAMESPACE
