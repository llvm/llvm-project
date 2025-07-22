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

namespace LIBC_NAMESPACE_DECL {
namespace internal {

// returns true if the character exists in the string
LIBC_INLINE static bool wcschr(wchar_t c, const wchar_t *str) {
  for (int n = 0; str[n]; ++n) {
    if (str[n] == c)
      return true;
  }
  return false;
}

// bool should be true for wcscspn for complimentary span
// should be false for wcsspn since we want it to span
LIBC_INLINE static size_t wcsspn(const wchar_t *s1, const wchar_t *s2,
                                 bool not_match_set) {
  size_t i = 0;
  for (; s1[i]; ++i) {
    bool in_set = wcschr(s1[i], s2);
    if (in_set == not_match_set)
      return i;
  }
  return i;
}

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL

#endif //  LLVM_LIBC_SRC_WCHAR_WCHAR_UTILS_H
