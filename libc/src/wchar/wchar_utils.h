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

LIBC_INLINE static const wchar_t *wcschr(const wchar_t *s, wchar_t c) {
  for (; *s && *s != c; ++s)
    ;
  return (*s == c) ? s : nullptr;
}

// bool should be true for wcscspn for complimentary span
// should be false for wcsspn since we want it to span
LIBC_INLINE static size_t wcsspn(const wchar_t *s1, const wchar_t *s2,
                                 bool not_match_set) {
  size_t i = 0;
  for (; s1[i]; ++i) {
    bool in_set = internal::wcschr(s2, s1[i]);
    if (in_set == not_match_set)
      return i;
  }
  return i;
}

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL

#endif //  LLVM_LIBC_SRC_WCHAR_WCHAR_UTILS_H
