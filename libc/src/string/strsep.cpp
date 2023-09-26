//===-- Implementation of strsep ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strsep.h"

#include "src/string/string_utils.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(char *, strsep, (char **stringp, const char *delim)) {
  if (!*stringp)
    return nullptr;
  return internal::string_token<false>(*stringp, delim, stringp);
}

} // namespace LIBC_NAMESPACE
