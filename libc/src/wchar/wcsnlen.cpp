//===-- Implementation of wcsnlen -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wchar/wcsnlen.h"

#include "hdr/types/size_t.h"
#include "hdr/types/wchar_t.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(size_t, wcsnlen, (const wchar_t *src, size_t maxlen)) {
  size_t i = 0;
  for (; i < maxlen && src[i]; ++i)
    ;
  return i;
}

} // namespace LIBC_NAMESPACE_DECL
