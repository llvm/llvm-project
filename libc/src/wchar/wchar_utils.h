//===-- wchar utils ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_WCHAR_WCHAR_UTILS_H
#define LLVM_LIBC_SRC_WCHAR_WCHAR_UTILS_H

#include "hdr/types/size_t.h"
#include "hdr/types/wchar_t.h"
#include "src/__support/common.h"
#include "src/__support/macros/attributes.h" // LIBC_INLINE

bool check_span(wchar_t c, const wchar_t *str) {
  for (int n = 0; str[n]; ++n) {
    if (str[n] == c)
      return true;
  }
  return false;
}

namespace LIBC_NAMESPACE_DECL {
namespace internal {

// To avoid duplicated code, call this with true for wcscspn and call with false
// for wcsspn
LIBC_INLINE size_t inline_wcsspn(const wchar_t *s1, const wchar_t *s2,
                                 bool invert) {
  size_t i = 0;
  for (; s1[i]; ++i) {
    bool check = check_span(s1[i], s2);
    check = invert ? !check : check;
    if (!check)
      return i;
  }
  return i;
}

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL

#endif //  LLVM_LIBC_SRC_WCHAR_WCHAR_UTILS_H
