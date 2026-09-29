//===-- Implementation of wcscat ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wchar/wcscat.h"

#include "hdr/types/size_t.h"
#include "hdr/types/wchar_t.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/string/string_utils.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(wchar_t *, wcscat,
                   (wchar_t *__restrict s1, const wchar_t *__restrict s2)) {
  size_t size_1 = internal::string_length(s1);
  size_t i = 0;
  for (; s2[i] != L'\0'; i++)
    s1[size_1 + i] = s2[i];
  s1[size_1 + i] = L'\0';
  return s1;
}

} // namespace LIBC_NAMESPACE_DECL
