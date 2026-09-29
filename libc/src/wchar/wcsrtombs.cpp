//===-- Implementation of wcsrtombs ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wchar/wcsrtombs.h"

#include "hdr/types/char32_t.h"
#include "hdr/types/mbstate_t.h"
#include "hdr/types/size_t.h"
#include "hdr/types/wchar_t.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/wchar/mbstate.h"
#include "src/__support/wchar/wcsnrtombs.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(size_t, wcsrtombs,
                   (char *__restrict s, const wchar_t **__restrict pwcs,
                    size_t n, mbstate_t *__restrict ps)) {
  LIBC_CRASH_ON_NULLPTR(pwcs);
  static internal::mbstate internal_mbstate;
  auto result = internal::wcsnrtombs(
      s, pwcs, SIZE_MAX, n,
      ps == nullptr ? &internal_mbstate
                    : reinterpret_cast<internal::mbstate *>(ps));
  if (!result.has_value()) {
    libc_errno = result.error();
    return -1;
  }

  return result.value();
}

} // namespace LIBC_NAMESPACE_DECL
