//===-- Implementation of wmemchr -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wchar/wmemchr.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(const wchar_t *, wmemchr,
                   (const wchar_t *s, wchar_t c, size_t n)) {
  for (size_t i = 0; i < n; i++) {
    if (s[i] == c) {
      return &s[i];
    }
  }

  return nullptr;
}

} // namespace LIBC_NAMESPACE_DECL
