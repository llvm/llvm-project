//===-- Implementation of wcrtomb -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wchar/wcrtomb.h"

#include "hdr/types/mbstate_t.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/wchar/wcrtomb.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(size_t, wcrtomb,
                   (char *__restrict s, wchar_t wc, mbstate_t *__restrict ps)) {
  static mbstate_t internal_mbstate{0, 0, 0};

  auto result =
      internal::wcrtomb(s, wc, ps == nullptr ? &internal_mbstate : ps);

  if (!result.has_value()) {
    libc_errno = EILSEQ;
    return -1;
  }

  return result.value();
}

} // namespace LIBC_NAMESPACE_DECL
