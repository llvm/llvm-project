//===-- Implementation of wcsrchr -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wchar/wcsrchr.h"

#include "hdr/types/wchar_t.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(const wchar_t *, wcsrchr, (const wchar_t *s, wchar_t c)) {
  LIBC_CRASH_ON_NULLPTR(s);

  const wchar_t *last_occurrence = nullptr;
  while (true) {
    if (*s == c)
      last_occurrence = s;
    if (*s == L'\0')
      return last_occurrence;
    ++s;
  }
}

} // namespace LIBC_NAMESPACE_DECL
