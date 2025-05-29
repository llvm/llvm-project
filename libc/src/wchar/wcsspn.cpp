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
#include "src/__support/CPP/bitset.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(size_t, wcsspn, (const wchar_t *s1, const wchar_t *s2)) {
  size_t i = 0;
  int in_s2 = 0;
  for (; s1[i]; ++i) {
    for (int n = 0; s2[n] && in_s2 == 0; ++n) {
      if (s1[i] == s2[n])
        in_s2 = 1;
    }
    if (in_s2 == 0) {
      return i;
    }
    in_s2 = 0;
  }
  return i;
}

} // namespace LIBC_NAMESPACE_DECL
