//===-- Implementation of wcsstr ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wchar/wcsstr.h"

#include "hdr/types/size_t.h"
#include "hdr/types/wchar_t.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/string/string_utils.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(const wchar_t *, wcsstr,
                   (const wchar_t *s1, const wchar_t *s2)) {
  size_t s1_len = internal::string_length(s1);
  size_t s2_len = internal::string_length(s2);
  if (s2_len == 0)
    return s1;
  if (s2_len > s1_len)
    return nullptr;
  for (size_t i = 0; i <= (s1_len - s2_len); ++i) {
    size_t j = 0;
    // j will increment until the characters don't match or end of string.
    for (; j < s2_len && s1[i + j] == s2[j]; ++j)
      ;
    if (j == s2_len)
      return (s1 + i);
  }
  return nullptr;
}

} // namespace LIBC_NAMESPACE_DECL
