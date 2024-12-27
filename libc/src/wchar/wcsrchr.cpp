//===-- Implementation of wcsrchr
//------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wchar/wcsrchr.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "wcslen.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(const wchar_t *, wcsrchr, (const wchar_t *s, wchar_t c)) {
  size_t length = wcslen(s);
  for (size_t i = 0; i < length; i++) {
    if (s[length - i] == c)
      return &s[length - i];
  }
  return nullptr;
}

} // namespace LIBC_NAMESPACE_DECL
