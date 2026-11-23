//===-- Implementation of wmemmove ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wchar/wmemmove.h"

#include "hdr/types/size_t.h"
#include "hdr/types/wchar_t.h"
#include "src/__support/common.h"
#include "src/__support/macros/null_check.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(wchar_t *, wmemmove,
                   (wchar_t * dest, const wchar_t *src, size_t n)) {
  LIBC_CRASH_ON_NULLPTR(dest);
  LIBC_CRASH_ON_NULLPTR(src);

  __builtin_memmove(dest, src, n * sizeof(wchar_t));
  return dest;
}

} // namespace LIBC_NAMESPACE_DECL
