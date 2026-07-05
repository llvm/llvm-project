//===-- Implementation of mbrlen ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wchar/mbrlen.h"

#include "hdr/types/mbstate_t.h"
#include "hdr/types/size_t.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/wchar/mbrtowc.h"
#include "src/__support/wchar/mbstate.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(size_t, mbrlen,
                   (const char *__restrict s, size_t n,
                    mbstate_t *__restrict ps)) {
  static internal::mbstate internal_mbstate;
  auto ret = internal::mbrtowc(nullptr, s, n,
                               ps == nullptr
                                   ? &internal_mbstate
                                   : reinterpret_cast<internal::mbstate *>(ps));
  if (!ret.has_value()) {
    // Encoding failure
    libc_errno = ret.error();
    return -1;
  }
  return ret.value();
}

} // namespace LIBC_NAMESPACE_DECL
