//===-- Implementation of strtod_l
//-----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/strtod_l.h"
#include "src/__support/common.h"
#include "src/stdlib/strtod.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(double, strtod_l,
                   (const char *__restrict str, char **__restrict str_end,
                    locale_t)) {
  return strtod(str, str_end);
}

} // namespace LIBC_NAMESPACE
