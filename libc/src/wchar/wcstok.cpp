//===-- Implementation of wcstok ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wchar/wcstok.h"

#include "hdr/types/wchar_t.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(wchar_t *, wcstok,
                   (wchar_t *__restrict str, const wchar_t *__restrict delim,
                    wchar_t **__restrict ptr)) {
  if (str == nullptr)
    str = *ptr;

  bool foundTokenStart = false;
  wchar_t *out = nullptr;
  wchar_t *str_ptr;
  for (str_ptr = str; *str_ptr != L'\0'; str_ptr++) {
    bool inDelim = false;
    for (const wchar_t *delim_ptr = delim; *delim_ptr != L'\0' && !inDelim;
         delim_ptr++)
      if (*str_ptr == *delim_ptr)
        inDelim = true;

    if (!inDelim && !foundTokenStart) {
      foundTokenStart = true;
      out = str_ptr;
    } else if (inDelim && foundTokenStart) {
      *str_ptr = L'\0';
      *ptr = str_ptr + 1;
      return out;
    }
  }

  *ptr = str_ptr;
  return out;
}

} // namespace LIBC_NAMESPACE_DECL
