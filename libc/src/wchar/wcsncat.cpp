//===-- Implementation of wcsncat -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wchar/wcsncat.h"

#include "hdr/types/size_t.h"
#include "hdr/types/wchar_t.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/string/string_utils.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(wchar_t *, wcsncat,
                   (wchar_t *__restrict s1, const wchar_t *__restrict s2,
                    size_t n)) {
  size_t size = internal::string_length(s1);
  size_t i = 0;
  for (; s2[i] && i < n; ++i)
    s1[size + i] = s2[i];
  // Appending null character to the end of the result.
  s1[size + i] = L'\0';
  return s1;
}

} // namespace LIBC_NAMESPACE_DECL
