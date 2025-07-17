//===-- Implementation of wcsnrtombs --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/wchar/wcsnrtombs.h"

#include "hdr/types/char32_t.h"
#include "hdr/types/size_t.h"
#include "hdr/types/wchar_t.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/wchar/mbstate.h"
#include "src/__support/wchar/string_converter.h"

namespace LIBC_NAMESPACE_DECL {
namespace internal {

ErrorOr<size_t> wcsnrtombs(char *__restrict s, const wchar_t **__restrict pwcs,
                           size_t nwc, size_t len, mbstate *ps) {
  CharacterConverter cr(ps);
  if (!cr.isValidState())
    return Error(EINVAL);

  if (s == nullptr)
    len = SIZE_MAX;

  StringConverter<char32_t> str_conv(reinterpret_cast<const char32_t *>(*pwcs),
                                     ps, len, nwc);
  size_t dst_idx = 0;
  ErrorOr<char8_t> converted = str_conv.popUTF8();
  while (converted.has_value()) {
    if (s != nullptr)
      s[dst_idx] = converted.value();

    if (converted.value() == '\0') {
      *pwcs = nullptr;
      return dst_idx;
    }

    dst_idx++;
    converted = str_conv.popUTF8();
  }

  *pwcs += str_conv.getSourceIndex();
  if (converted.error() == -1) // if we hit conversion limit
    return dst_idx;

  return Error(converted.error());
}
} // namespace internal
} // namespace LIBC_NAMESPACE_DECL
