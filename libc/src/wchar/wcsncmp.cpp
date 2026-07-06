//===-- Implementation of wcsncmp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wchar/wcsncmp.h"

#include "hdr/types/size_t.h"
#include "hdr/types/wchar_t.h"
#include "src/__support/common.h"
#include "src/__support/macros/null_check.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, wcsncmp,
                   (const wchar_t *left, const wchar_t *right, size_t n)) {
  LIBC_CRASH_ON_NULLPTR(left);
  LIBC_CRASH_ON_NULLPTR(right);

  if (n == 0)
    return 0;

  auto comp = [](wchar_t l, wchar_t r) -> int { return l - r; };

  for (; n > 1; --n, ++left, ++right) {
    wchar_t lc = *left;
    if (!comp(lc, '\0') || comp(lc, *right))
      break;
  }
  return comp(*left, *right);
}

} // namespace LIBC_NAMESPACE_DECL
