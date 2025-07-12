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

bool isADelimeter(wchar_t wc, const wchar_t *delimiters) {
  for (const wchar_t *delim_ptr = delimiters; *delim_ptr != L'\0'; ++delim_ptr)
    if (wc == *delim_ptr)
      return true;
  return false;
}

LLVM_LIBC_FUNCTION(wchar_t *, wcstok,
                   (wchar_t *__restrict str, const wchar_t *__restrict delim,
                    wchar_t **__restrict context)) {
  if (str == nullptr) {
    if (*context == nullptr)
      return nullptr;

    str = *context;
  }

  wchar_t *tok_start, *tok_end;
  for (tok_start = str; *tok_start != L'\0' && isADelimeter(*tok_start, delim);
       ++tok_start)
    ;

  for (tok_end = tok_start; *tok_end != L'\0' && !isADelimeter(*tok_end, delim);
       ++tok_end)
    ;

  if (*tok_end != L'\0') {
    *tok_end = L'\0';
    ++tok_end;
  }
  *context = tok_end;
  return *tok_start == L'\0' ? nullptr : tok_start;
}

} // namespace LIBC_NAMESPACE_DECL
