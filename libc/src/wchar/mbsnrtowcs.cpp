//===-- Implementation of mbsnrtowcs --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wchar/mbsnrtowcs.h"

#include "hdr/types/size_t.h"
#include "hdr/types/wchar_t.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/wchar/mbsnrtowcs.h"
#include "src/__support/wchar/mbstate.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(size_t, mbsnrtowcs,
                   (wchar_t *__restrict dst, const char **__restrict src,
                    size_t nmc, size_t len, mbstate_t *__restrict ps)) {
  static internal::mbstate internal_mbstate;
  // If destination is null, ignore len
  len = dst == nullptr ? SIZE_MAX : len;
  auto ret = internal::mbsnrtowcs(
      dst, src, nmc, len,
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
