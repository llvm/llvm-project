//===-- Implementation of wcsrtombs ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/wchar/wcsrtombs.h"
#include "src/__support/error_or.h"
#include "src/__support/wchar/character_converter.h"
#include "src/__support/wchar/mbstate.h"
#include "src/__support/wchar/wcrtomb.h"

#include "hdr/types/char32_t.h"
#include "hdr/types/size_t.h"
#include "hdr/types/wchar_t.h"
#include "src/__support/common.h"
#include "src/__support/libc_assert.h"

namespace LIBC_NAMESPACE_DECL {
namespace internal {

ErrorOr<size_t> wcsrtombs(char *__restrict dst, const wchar_t **__restrict src,
                          size_t len, mbstate *__restrict ps) {
  static_assert(sizeof(wchar_t) == 4);

  if (dst == nullptr)
    return Error(-1);

  size_t bytes_written = 0;
  const wchar_t *wc_ptr = *src;
  while (bytes_written < len) {
    char buf[4];
    auto result = internal::wcrtomb(dst + bytes_written, *wc_ptr, ps,
                                    len - bytes_written);
    if (!result.has_value())
      return result; // forward the error

    if (result.value() == -1) // couldn't complete the conversion
      return len;

    // terminate the loop after converting the null wide character
    if (*wc_ptr == L'\0') {
      *src = '\0';
      return bytes_written;
    }

    bytes_written += result.value();
    wc_ptr++;
  }

  return bytes_written;
}

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL
