//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of towupper.
///
//===----------------------------------------------------------------------===//

#include "src/wctype/towupper.h"
#include "hdr/types/wint_t.h"
#include "src/__support/common.h"
#include "src/__support/wctype_utils.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(wint_t, towupper, (wint_t c)) {
  if (c == static_cast<wint_t>(static_cast<wchar_t>(c)))
    return static_cast<wint_t>(internal::toupper(static_cast<wchar_t>(c)));
  return c;
}

} // namespace LIBC_NAMESPACE_DECL
