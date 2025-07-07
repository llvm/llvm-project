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

template <typename Check>
LIBC_INLINE size_t inline_wcsspn(const wchar_t *s1, Check check) {
  size_t i = 0;
  for (; s1[i]; ++i) {
    if (!check(s1[i]))
      return i;
  }
  return i;
}

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL

#endif //  LLVM_LIBC_SRC_WCHAR_WCHAR_UTILS_H
