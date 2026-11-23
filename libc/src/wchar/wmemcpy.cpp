//===-- Implementation of wmemcpy -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wchar/wmemcpy.h"

#include "hdr/types/size_t.h"
#include "hdr/types/wchar_t.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/string/memory_utils/inline_memcpy.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(wchar_t *, wmemcpy,
                   (wchar_t *__restrict s1, const wchar_t *__restrict s2,
                    size_t n)) {
  inline_memcpy(s1, s2, n * sizeof(wchar_t));
  return s1;
}

} // namespace LIBC_NAMESPACE_DECL
