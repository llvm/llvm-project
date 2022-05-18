//===-- Implementation of strlcpy -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strlcpy.h"
#include "src/string/bzero.h"
#include "src/string/memory_utils/memcpy_implementations.h"
#include "src/string/memory_utils/memset_implementations.h"
#include "src/string/string_utils.h"

#include "src/__support/common.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(size_t, strlcpy,
                   (char *__restrict dst, const char *__restrict src,
                    size_t size)) {
  size_t len = internal::string_length(src);
  if (!size)
    return len;
  size_t n = len < size - 1 ? len : size - 1;
  inline_memcpy(dst, src, n);
  inline_memset(dst + n, 0, size - n);
  return len;
}

} // namespace __llvm_libc
