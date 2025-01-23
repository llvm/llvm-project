//===-- Wide String utils -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_WCHAR_WIDE_STRING_UTILS_H
#define LLVM_LIBC_SRC_WCHAR_WIDE_STRING_UTILS_H

#include "src/__support/macros/config.h"
#include "hdr/types/size_t.h"
#include "hdr/types/wchar_t.h"

namespace LIBC_NAMESPACE_DECL {
namespace internal {

LIBC_INLINE size_t wide_string_length(const wchar_t *src) {
  const wchar_t *cpy = src;
  while (*cpy)
    ++cpy;
  return cpy - src;
}

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL

#endif //  LLVM_LIBC_SRC_WCHAR_WIDE_STRING_UTILS_H
