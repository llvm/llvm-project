//===-- Implementation of towlower ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wctype/towlower.h"
#include "src/__support/common.h"
#include "src/__support/wctype_utils.h"

#include "hdr/types/wint_t.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(wint_t, towlower, (wint_t c)) {
  return internal::tolower(static_cast<wchar_t>(c));
}

} // namespace LIBC_NAMESPACE_DECL
