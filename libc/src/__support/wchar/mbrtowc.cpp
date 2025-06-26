//===-- Implementation for mbrtowc function ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/wchar/mbrtowc.h"
#include "hdr/errno_macros.h"
#include "hdr/types/mbstate_t.h"
#include "hdr/types/size_t.h"
#include "hdr/types/wchar_t.h"
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"
#include "src/__support/wchar/character_converter.h"
#include "src/__support/wchar/mbstate.h"

namespace LIBC_NAMESPACE_DECL {
namespace internal {

ErrorOr<size_t> mbrtowc(wchar_t *__restrict pwc, const char *__restrict s,
                        size_t n, mbstate *__restrict ps) {
  CharacterConverter char_conv(ps);
  if (!char_conv.isValidState())
    return Error(EINVAL);
  if (s == nullptr)
    return 0;
  size_t i = 0;
  // Reading in bytes until we have a complete wc or error
  for (; i < n && !char_conv.isFull(); ++i) {
    int err = char_conv.push(static_cast<char8_t>(s[i]));
    // Encoding error
    if (err == -1)
      return Error(EILSEQ);
  }
  auto wc = char_conv.pop_utf32();
  if (wc.has_value()) {
    *pwc = wc.value();
    // null terminator -> return 0
    if (wc.value() == L'\0')
      return 0;
    return i;
  }
  // Incomplete but potentially valid
  return -2;
}

} // namespace internal

} // namespace LIBC_NAMESPACE_DECL
