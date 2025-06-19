//===-- Implementation of wcspbrk -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wchar/wcspbrk.h"

#include "hdr/types/wchar_t.h"
#include "src/__support/common.h"
#include "src/__support/macros/null_check.h"

namespace LIBC_NAMESPACE_DECL {

bool contains_char(const wchar_t *str, wchar_t target) {
  for (; *str != L'\0'; str++)
    if (*str == target)
      return true;

  return false;
}

LLVM_LIBC_FUNCTION(const wchar_t *, wcspbrk,
                   (const wchar_t *src, const wchar_t *breakset)) {
  LIBC_CRASH_ON_NULLPTR(src);
  LIBC_CRASH_ON_NULLPTR(breakset);

  // currently O(n * m), can be further optimized to O(n + m) with a hash set
  for (int src_idx = 0; src[src_idx] != 0; src_idx++)
    if (contains_char(breakset, src[src_idx]))
      return src + src_idx;

  return nullptr;
}

} // namespace LIBC_NAMESPACE_DECL
