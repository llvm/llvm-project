//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the wcwidth function implementation.
///
//===----------------------------------------------------------------------===//

#include "src/wchar/wcwidth.h"

#include "hdr/types/wchar_t.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/wctype_utils.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, wcwidth, (wchar_t wc)) {
  if (wc == L'\0')
    return 0;
  // In LIBC_WCTYPE_MODE_UTF8, wctype_utils does not yet distinguish zero-width
  // combining marks or double-width characters, so all printable characters
  // report a column width of 1.
  return internal::isprint(wc) ? 1 : -1;
}

} // namespace LIBC_NAMESPACE_DECL
