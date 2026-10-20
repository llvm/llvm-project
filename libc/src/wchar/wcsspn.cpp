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
#include "src/__support/macros/null_check.h"
#include "wchar_utils.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(size_t, wcsspn, (const wchar_t *s1, const wchar_t *s2)) {
  LIBC_CRASH_ON_NULLPTR(s1);
  LIBC_CRASH_ON_NULLPTR(s2);
  return internal::wcsspn(s1, s2, /*not_match_set=*/false);
}

} // namespace LIBC_NAMESPACE_DECL
