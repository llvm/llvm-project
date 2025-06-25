//===-- Implementation for mbsrtowcs function -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/wchar/mbsrtowcs.h"
#include "hdr/types/mbstate_t.h"
#include "hdr/types/size_t.h"
#include "hdr/types/wchar_t.h"
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"
#include "src/__support/wchar/mbrtowc.h"
#include "src/__support/wchar/mbstate.h"

namespace LIBC_NAMESPACE_DECL {
namespace internal {

ErrorOr<size_t> mbsrtowcs(wchar_t *__restrict dst, const char **__restrict src,
                          size_t len, mbstate *__restrict ps) {
  size_t i = 0;
  // Converting characters until we reach error or null terminator
  for (; i < len; ++i, ++dst) {
    auto check = mbrtowc(dst, *src, 4, ps);
    // Encoding error/invalid mbstate
    if (!check.has_value())
      return Error(check.error());
    // Successfully encoded, check for null terminator
    if (*dst == L'\0') {
      *src = nullptr;
      return i;
    }
    // Set src to point right after the last character converted
    *src = *src + check.value();
  }
  return i;
}

} // namespace internal

} // namespace LIBC_NAMESPACE_DECL
