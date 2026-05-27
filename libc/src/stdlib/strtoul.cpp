//===-- Implementation of strtoul -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/strtoul.h"
#include "src/__support/macros/config.h"
#include "src/stdlib/str_to_util.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(unsigned long, strtoul,
                   (const char *__restrict str, char **__restrict str_end,
                    int base)) {
  return internal::str_to_helper<unsigned long>(str, str_end, base);
}

} // namespace LIBC_NAMESPACE_DECL
