//===-- Implementation of strcat ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strcat.h"
#include "src/string/strcpy.h"
#include "src/string/string_utils.h"

#include "src/__support/common.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(char *, strcat,
                   (char *__restrict dest, const char *__restrict src)) {
  size_t dest_length = internal::string_length(dest);
  size_t src_length = internal::string_length(src);
  LIBC_NAMESPACE::strcpy(dest + dest_length, src);
  dest[dest_length + src_length] = '\0';
  return dest;
}

} // namespace LIBC_NAMESPACE
