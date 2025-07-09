//===-- Implementation of wcsncpy -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wchar/wcsncpy.h"

#include "hdr/types/size_t.h"
#include "hdr/types/wchar_t.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(wchar_t *, wcsncpy,
                   (wchar_t *__restrict s1, const wchar_t *__restrict s2,
                    size_t n)) {
  size_t i = 0;
  // Copy up until \0 is found.
  for (; i < n && s2[i] != L'\0'; ++i)
    s1[i] = s2[i];
  // When s2 is shorter than n, append \0.
  for (; i < n; ++i)
    s1[i] = L'\0';
  return s1;
}

} // namespace LIBC_NAMESPACE_DECL
