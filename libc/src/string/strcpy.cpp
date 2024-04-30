//===-- Implementation of strcpy ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strcpy.h"
#include "src/string/memory_utils/inline_memcpy.h"
#include "src/string/string_utils.h"

#include "src/__support/common.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(char *, strcpy,
                   (char *__restrict dest, const char *__restrict src)) {
  size_t size = internal::string_length(src) + 1;
  inline_memcpy(dest, src, size);
  return dest;
}

} // namespace LIBC_NAMESPACE
