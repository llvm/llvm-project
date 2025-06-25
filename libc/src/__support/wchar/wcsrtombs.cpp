//===-- Implementation of wcsrtombs ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/wchar/wcsrtombs.h"
#include "hdr/types/char32_t.h"
#include "hdr/types/size_t.h"
#include "hdr/types/wchar_t.h"
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/libc_assert.h"
#include "src/__support/wchar/character_converter.h"
#include "src/__support/wchar/mbstate.h"
#include "src/__support/wchar/wcrtomb.h"

namespace LIBC_NAMESPACE_DECL {
namespace internal {

ErrorOr<size_t> wcsrtombs(char *__restrict dst, const wchar_t **__restrict src,
                          size_t len, mbstate *__restrict ps) {
  if (src == nullptr)
    return Error(-1);

  // ignore len parameter when theres no destination string
  if (dst == nullptr)
    len = SIZE_MAX;

  size_t bytes_written = 0;
  while (bytes_written < len) {
    auto result =
        internal::wcrtomb(dst == nullptr ? nullptr : dst + bytes_written, **src,
                          ps, len - bytes_written);
    if (!result.has_value())
      return result; // forward the error

    // couldn't complete conversion
    if (result.value() == static_cast<size_t>(-1))
      return len;

    // terminate the loop after converting the null wide character
    if (**src == L'\0') {
      *src = nullptr;
      return bytes_written;
    }

    bytes_written += result.value();
    (*src)++;
  }

  return bytes_written;
}

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL
