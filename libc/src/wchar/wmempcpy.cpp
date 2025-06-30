//===-- Implementation of wmempcpy ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wchar/wmempcpy.h"

#include "hdr/types/size_t.h"
#include "hdr/types/wchar_t.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(wchar_t *, wmempcpy,
                   (wchar_t *__restrict to, const wchar_t *__restrict from,
                    size_t size)) {
  __builtin_memcpy(to, from, size * sizeof(wchar_t));
  return reinterpret_cast<wchar_t *>(to) + size;
}

} // namespace LIBC_NAMESPACE_DECL
