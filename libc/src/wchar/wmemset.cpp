//===-- Implementation of wmemset -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wchar/wmemset.h"

#include "hdr/types/size_t.h"
#include "hdr/types/wchar_t.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(wchar_t *, wmemset, (wchar_t * s, wchar_t c, size_t n)) {
  for (size_t i = 0; i < n; i++)
    s[i] = c;

  return s;
}

} // namespace LIBC_NAMESPACE_DECL
