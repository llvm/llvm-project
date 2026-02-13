//===-- Implementation of mbstowcs ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/mbstowcs.h"

#include "hdr/types/size_t.h"
#include "hdr/types/wchar_t.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/__support/wchar/mbsnrtowcs.h"
#include "src/__support/wchar/mbstate.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(size_t, mbstowcs,
                   (wchar_t *__restrict pwcs, const char *__restrict s,
                    size_t n)) {
  LIBC_CRASH_ON_NULLPTR(s);
  // If destination is null, ignore n
  n = pwcs == nullptr ? SIZE_MAX : n;
  static internal::mbstate internal_mbstate;
  const char *temp = s;
  auto ret = internal::mbsnrtowcs(pwcs, &temp, SIZE_MAX, n, &internal_mbstate);

  if (!ret.has_value()) {
    // Encoding failure
    libc_errno = ret.error();
    return -1;
  }
  return ret.value();
}

} // namespace LIBC_NAMESPACE_DECL
