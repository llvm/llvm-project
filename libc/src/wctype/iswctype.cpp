//===-- Implementation of iswctype ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wctype/iswctype.h"
#include "src/__support/common.h"
#include "src/__support/wctype_utils.h"

#include "hdr/types/wctype_t.h"
#include "hdr/types/wint_t.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, iswctype, (wint_t c, wctype_t desc)) {
  return internal::iswctype(static_cast<wchar_t>(c), desc);
}

} // namespace LIBC_NAMESPACE_DECL
