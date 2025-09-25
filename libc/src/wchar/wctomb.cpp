//===-- Implementation of wctomb ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wchar/wctomb.h"

#include "hdr/types/wchar_t.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/wchar/mbstate.h"
#include "src/__support/wchar/wcrtomb.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, wctomb, (char *s, wchar_t wc)) {
  static internal::mbstate internal_mbstate;
  if (s == nullptr)
    return 0;

  auto result = internal::wcrtomb(s, wc, &internal_mbstate);

  if (!result.has_value()) { // invalid wide character
    libc_errno = EILSEQ;
    return -1;
  }

  return static_cast<int>(result.value());
}

} // namespace LIBC_NAMESPACE_DECL
