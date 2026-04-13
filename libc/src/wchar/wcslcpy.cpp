//===-- Implementation of wcslcpy -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wchar/wcslcpy.h"

#include "hdr/types/size_t.h"
#include "hdr/types/wchar_t.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/string/string_utils.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(size_t, wcslcpy,
                   (wchar_t *__restrict dst, const wchar_t *__restrict src,
                    size_t dstsize)) {
  size_t len = internal::string_length(src);
  if (dstsize == 0)
    return len;
  size_t i = 0;
  for (; i < dstsize - 1 && src[i] != L'\0'; ++i)
    dst[i] = src[i];
  dst[i] = L'\0';
  return len;
}

} // namespace LIBC_NAMESPACE_DECL
