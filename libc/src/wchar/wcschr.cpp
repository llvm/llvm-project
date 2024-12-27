//===-- Implementation of wcschr ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wchar/wcschr.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "wcslen.h"
#include "wmemchr.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(const wchar_t *, wcschr, (const wchar_t *s, wchar_t c)) {
  return wmemchr(s, c, wcslen(s));
}

} // namespace LIBC_NAMESPACE_DECL
