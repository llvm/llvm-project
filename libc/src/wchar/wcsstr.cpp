//===-- Implementation of wcsstr ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wchar/wcsstr.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "wcslen.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(const wchar_t *, wcsstr,
                   (const wchar_t *s, const wchar_t *needle)) {
  size_t s_len = wcslen(s);
  size_t needle_len = wcslen(needle);

  if (needle_len > s_len)
    return nullptr;

  for (size_t i = 0; i < s_len; i++) {
    size_t end = needle_len + i;
    if (end > s_len)
      break;

    bool found = true;
    for (size_t x = 0; x < needle_len; x++) {
      if (s[i + x] != needle[x]) {
        found = false;
        break;
      }
    }

    if (!found)
      continue;
    return &s[i];
  }
  return nullptr;
}

} // namespace LIBC_NAMESPACE_DECL
