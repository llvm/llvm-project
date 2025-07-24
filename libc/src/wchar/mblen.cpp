//===-- Implementation of mblen -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wchar/mblen.h"

#include "hdr/types/size_t.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/wchar/mbrtowc.h"
#include "src/__support/wchar/mbstate.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, mblen, (const char *s, size_t n)) {
  // returns 0 since UTF-8 encoding is not state-dependent
  if (s == nullptr)
    return 0;
  internal::mbstate internal_mbstate;
  auto ret = internal::mbrtowc(nullptr, s, n, &internal_mbstate);
  if (!ret.has_value() || static_cast<int>(ret.value()) == -2) {
    // Encoding failure
    if (!ret.has_value())
      libc_errno = EILSEQ;
    return -1;
  }
  return static_cast<int>(ret.value());
}

} // namespace LIBC_NAMESPACE_DECL
