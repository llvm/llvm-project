//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the wcscasecmp function implementation.
///
//===----------------------------------------------------------------------===//

#include "src/wchar/wcscasecmp.h"

#include "hdr/types/wchar_t.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/__support/wctype_utils.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, wcscasecmp,
                   (const wchar_t *left, const wchar_t *right)) {
  LIBC_CRASH_ON_NULLPTR(left);
  LIBC_CRASH_ON_NULLPTR(right);

  for (;; ++left, ++right) {
    wchar_t lc = *left;
    if (lc == L'\0' || internal::tolower(lc) != internal::tolower(*right))
      break;
  }
  return internal::threeway_cmp_single(internal::tolower(*left),
                                       internal::tolower(*right));
}

} // namespace LIBC_NAMESPACE_DECL
