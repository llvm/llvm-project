//===-- Implementation of wcpcpy ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wchar/wcpcpy.h"

#include "hdr/types/size_t.h"
#include "hdr/types/wchar_t.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/string/string_utils.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(wchar_t *, wcpcpy,
                   (wchar_t *__restrict s1, const wchar_t *__restrict s2)) {
  size_t size = internal::string_length(s2);
  inline_memcpy(s1, s2, (size + 1) * sizeof(wchar_t));
  wchar_t *result = s1 + size;
  return result;
}

} // namespace LIBC_NAMESPACE_DECL
