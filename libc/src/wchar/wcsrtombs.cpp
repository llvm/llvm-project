//===-- Implementation of wcsrtombs ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wchar/wcsrtombs.h"

#include "hdr/types/mbstate_t.h"
#include "hdr/types/wchar_t.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/__support/wchar/mbstate.h"
#include "src/__support/wchar/wcsrtombs.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(size_t, wcsrtombs,
                   (char *__restrict dst, const wchar_t **__restrict src,
                    size_t len, mbstate_t *__restrict ps)) {
  static internal::mbstate internal_mbstate;

  LIBC_CRASH_ON_NULLPTR(src);
  char buf[len];
  if (dst == nullptr)
    dst = buf;

  auto result = internal::wcsrtombs(
      dst, src, len,
      ps == nullptr ? &internal_mbstate
                    : reinterpret_cast<internal::mbstate *>(ps));

  if (!result.has_value()) {
    libc_errno = EILSEQ;
    return -1;
  }

  return result.value();
}

} // namespace LIBC_NAMESPACE_DECL
