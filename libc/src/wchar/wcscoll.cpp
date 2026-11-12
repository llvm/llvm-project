//===-- Implementation of wcscoll -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wchar/wcscoll.h"

#include "hdr/types/wchar_t.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"

namespace LIBC_NAMESPACE_DECL {

// TODO: Add support for locales.
LLVM_LIBC_FUNCTION(int, wcscoll, (const wchar_t *s1, const wchar_t *s2)) {
  LIBC_CRASH_ON_NULLPTR(s1);
  LIBC_CRASH_ON_NULLPTR(s2);

  for (; *s1 && (*s1 == *s2); ++s1, ++s2)
    ;

  return *s1 - *s2;
}

} // namespace LIBC_NAMESPACE_DECL
