//===-- Implementation of wcrtomb -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/wchar/wcrtomb.h"
#include "src/__support/error_or.h"
#include "src/__support/wchar/character_converter.h"
#include "src/__support/wchar/mbstate.h"

#include "hdr/errno_macros.h"
#include "hdr/types/char32_t.h"
#include "hdr/types/size_t.h"
#include "hdr/types/wchar_t.h"
#include "src/__support/common.h"
#include "src/__support/libc_assert.h"

namespace LIBC_NAMESPACE_DECL {
namespace internal {

ErrorOr<size_t> wcrtomb(char *__restrict s, wchar_t wc,
                        mbstate *__restrict ps) {
  static_assert(sizeof(wchar_t) == 4);

  CharacterConverter cr(ps);

  if (!cr.isValidState())
    return Error(EINVAL);

  int status = cr.push(static_cast<char32_t>(wc));
  if (status != 0)
    return Error(status);

  size_t count = 0;
  while (!cr.isEmpty()) {
    auto utf8 = cr.pop_utf8(); // can never fail as long as the push succeeded
    LIBC_ASSERT(utf8.has_value());

    *s = utf8.value();
    s++;
    count++;
  }
  return count;
}

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL
