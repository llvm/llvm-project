//===-- Implementation of strxfrm -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strxfrm.h"
#include "src/string/memory_utils/inline_memcpy.h"
#include "src/string/string_utils.h"

#include "src/__support/common.h"

namespace LIBC_NAMESPACE {

// TODO: Add support for locales.
LLVM_LIBC_FUNCTION(size_t, strxfrm,
                   (char *__restrict dest, const char *__restrict src,
                    size_t n)) {
  size_t len = internal::string_length(src);
  if (n > len)
    inline_memcpy(dest, src, len + 1);
  return len;
}

} // namespace LIBC_NAMESPACE
