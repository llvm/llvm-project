//===-- Collection of utils for implementing wide char functions --*-C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_WCTYPE_UTILS_H
#define LLVM_LIBC_SRC___SUPPORT_WCTYPE_UTILS_H

#include "src/__support/CPP/optional.h"
#include "src/__support/macros/attributes.h" // LIBC_INLINE

#define __need_wint_t
#define __need_wchar_t
#include <stddef.h> // needed for wint_t and wchar_t

namespace __llvm_libc {
namespace internal {

// ------------------------------------------------------
// Rationale: Since these classification functions are
// called in other functions, we will avoid the overhead
// of a function call by inlining them.
// ------------------------------------------------------

LIBC_INLINE cpp::optional<int> wctob(wint_t c) {
  // This needs to be translated to EOF at the callsite. This is to avoid
  // including stdio.h in this file.
  if (c > 127 || c < 0)
    return cpp::nullopt;
  return static_cast<int>(c);
}

LIBC_INLINE cpp::optional<wint_t> btowc(int c) {
  if (c > 127 || c < 0)
    return cpp::nullopt;
  return static_cast<wint_t>(c);
}

} // namespace internal
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC___SUPPORT_WCTYPE_UTILS_H
