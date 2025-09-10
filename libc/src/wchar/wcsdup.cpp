//===-- Implementation of wcsdup -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wchar/wcsdup.h"
#include "hdr/types/wchar_t.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/string/allocating_string_utils.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(wchar_t *, wcsdup, (const wchar_t *wcs)) {
  auto dup = internal::strdup(wcs);
  if (dup)
    return *dup;
  if (wcs != nullptr)
    libc_errno = ENOMEM;
  return nullptr;
}

} // namespace LIBC_NAMESPACE_DECL
