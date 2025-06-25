//===-- Implementation of mbsrtowcs ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wchar/mbsrtowcs.h"

#include "hdr/types/mbstate_t.h"
#include "hdr/types/size_t.h"
#include "hdr/types/wchar_t.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/__support/wchar/mbsrtowcs.h"
#include "src/__support/wchar/mbstate.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(size_t, mbsrtowcs,
                   (wchar_t *__restrict dst, const char **__restrict src,
                    size_t len, mbstate_t *__restrict ps)) {
  LIBC_CRASH_ON_NULLPTR(src);
  static internal::mbstate internal_mbstate;
  wchar_t temp[len];
  auto ret = internal::mbsrtowcs(
      dst == nullptr ? temp : dst, src, dst == nullptr ? SIZE_MAX : len,
      ps == nullptr ? &internal_mbstate
                    : reinterpret_cast<internal::mbstate *>(ps));
  if (!ret.has_value()) {
    // Encoding failure
    libc_errno = ret.error();
    return -1;
  }
  return ret.value();
}

} // namespace LIBC_NAMESPACE_DECL
