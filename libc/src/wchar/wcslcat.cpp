//===-- Implementation of wcslcat -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wchar/wcslcat.h"

#include "hdr/types/size_t.h"
#include "hdr/types/wchar_t.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/string/string_utils.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(size_t, wcslcat,
                   (wchar_t *__restrict dst, const wchar_t *__restrict src,
                    size_t dstsize)) {
  const size_t dstlen = internal::string_length(dst);
  const size_t srclen = internal::string_length(src);
  int limit = static_cast<int>(dstsize - dstlen - 1);
  size_t returnval = (dstsize < dstlen ? dstsize : dstlen) + srclen;
  if (limit < 0)
    return returnval;
  int i = 0;
  for (; i < limit && src[i] != L'\0'; ++i) {
    dst[dstlen + i] = src[i];
  }

  // appending null terminator if there is room
  if (dstlen + i < dstlen + dstsize)
    dst[dstlen + i] = L'\0';
  return returnval;
}

} // namespace LIBC_NAMESPACE_DECL
