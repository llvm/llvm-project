//===-- Implementation of wcpncpy -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wchar/wcpncpy.h"

#include "hdr/types/size_t.h"
#include "hdr/types/wchar_t.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(wchar_t *, wcpncpy,
                   (wchar_t *__restrict s1, const wchar_t *__restrict s2,
                    size_t n)) {
  if (n) {
    LIBC_CRASH_ON_NULLPTR(s1);
    LIBC_CRASH_ON_NULLPTR(s2);
  }
  size_t i;
  // Copy up until \0 is found.
  for (i = 0; i < n && s2[i] != '\0'; ++i)
    s1[i] = s2[i];
  // When n>strlen(src), n-strlen(src) \0 are appended.
  for (; i < n; ++i)
    s1[i] = L'\0';
  return s1 + i;
}

} // namespace LIBC_NAMESPACE_DECL
