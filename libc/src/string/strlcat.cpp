//===-- Implementation of strlcat -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strlcat.h"
#include "src/__support/CPP/span.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/string/string_utils.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(size_t, strlcat,
                   (char *__restrict dst, const char *__restrict src,
                    size_t size)) {
  cpp::span<char> dst_span(dst, size);
  size_t first_len = internal::strnlen(dst_span);
  if (first_len == dst_span.size())
    return dst_span.size() + internal::string_length(src);
  return first_len + internal::strlcpy(dst_span.subspan(first_len), src);
}

} // namespace LIBC_NAMESPACE_DECL
