//===-- Implementation of strncat -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strncat.h"
#include "src/string/string_utils.h"
#include "src/string/strncpy.h"

#include "src/__support/common.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(char *, strncat,
                   (char *__restrict dest, const char *__restrict src,
                    size_t count)) {
  size_t src_length = internal::string_length(src);
  size_t copy_amount = src_length > count ? count : src_length;
  size_t dest_length = internal::string_length(dest);
  LIBC_NAMESPACE::strncpy(dest + dest_length, src, copy_amount);
  dest[dest_length + copy_amount] = '\0';
  return dest;
}

} // namespace LIBC_NAMESPACE
