//===-- Implementation of wmemcmp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wchar/wmemcmp.h"

#include "hdr/types/size_t.h"
#include "hdr/types/wchar_t.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h" // LIBC_CRASH_ON_NULLPTR

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, wmemcmp,
                   (const wchar_t *s1, const wchar_t *s2, size_t n)) {
  LIBC_CRASH_ON_NULLPTR(s1);
  LIBC_CRASH_ON_NULLPTR(s2);
  for (size_t i = 0; i < n; ++i) {
    if (s1[i] != s2[i])
      return (int)(s1[i] - s2[i]);
  }
  // If it reaches the end, all n values must be the same.
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
