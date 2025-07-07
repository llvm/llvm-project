//===-- Implementation of wcsspn ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wchar/wcsspn.h"

#include "hdr/types/size_t.h"
#include "hdr/types/wchar_t.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "wchar_utils.h"

struct CheckSpan {
  const wchar_t *str;
  CheckSpan(const wchar_t *w) { str = w; }
  bool operator()(wchar_t c) {
    for (int n = 0; str[n]; ++n) {
      if (str[n] == c)
        return true;
    }
    return false;
  }
};

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(size_t, wcsspn, (const wchar_t *s1, const wchar_t *s2)) {
  CheckSpan check(s2);
  return internal::inline_wcsspn(s1, check);
}

} // namespace LIBC_NAMESPACE_DECL
