//===-- Implementation of wcscmp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wchar/wcscmp.h"

#include "hdr/types/size_t.h"
#include "hdr/types/wchar_t.h"
#include "src/__support/common.h"
#include "src/__support/macros/null_check.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, wcscmp, (const wchar_t *left, const wchar_t *right)) {
  LIBC_CRASH_ON_NULLPTR(left);
  LIBC_CRASH_ON_NULLPTR(right);

  for (; *left && (*left == *right); ++left, ++right)
    ;

  return *left - *right;
}

} // namespace LIBC_NAMESPACE_DECL
